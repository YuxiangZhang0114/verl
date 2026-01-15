import openai
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional


class DisagreementCalculator:
    """
    基于 vLLM OpenAI 兼容 API 的异构模型分歧计算器。
    用于计算学生模型和教师模型对同一段文本的 LogProb 差异。
    """
    
    def __init__(
        self, 
        student_base_url: str,
        student_model: str,
        student_tokenizer_path: str,
        teacher_base_url: str,
        teacher_model: str,
        teacher_tokenizer_path: str,
        api_key: str = "EMPTY"
    ):
        """
        初始化分歧计算器。
        
        Args:
            student_base_url: 学生模型 vLLM 服务地址，如 "http://localhost:8000/v1"
            student_model: 学生模型在 vLLM 中的名称
            student_tokenizer_path: 学生模型 tokenizer 路径（用于 chat template 和特殊 token）
            teacher_base_url: 教师模型 vLLM 服务地址
            teacher_model: 教师模型在 vLLM 中的名称
            teacher_tokenizer_path: 教师模型 tokenizer 路径
            api_key: API 密钥，vLLM 默认使用 "EMPTY"
        """
        # 创建 OpenAI 客户端
        print(f"Connecting to Student vLLM: {student_base_url} (model: {student_model})")
        self.stu_client = openai.OpenAI(base_url=student_base_url, api_key=api_key)
        self.stu_model = student_model
        
        print(f"Connecting to Teacher vLLM: {teacher_base_url} (model: {teacher_model})")
        self.tea_client = openai.OpenAI(base_url=teacher_base_url, api_key=api_key)
        self.tea_model = teacher_model
        
        # 加载 tokenizer（用于 chat template 和获取特殊 token 列表）
        print(f"Loading Student Tokenizer: {student_tokenizer_path}")
        self.stu_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path)
        self.stu_special_tokens = set(self.stu_tokenizer.all_special_tokens)
        print(f"  Special tokens: {self.stu_special_tokens}")
        
        print(f"Loading Teacher Tokenizer: {teacher_tokenizer_path}")
        self.tea_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_path)
        self.tea_special_tokens = set(self.tea_tokenizer.all_special_tokens)
        print(f"  Special tokens: {self.tea_special_tokens}")

    def _get_context_token_count(
        self, 
        client: openai.OpenAI, 
        model: str, 
        tokenizer: AutoTokenizer,
        context_messages: List[Dict[str, str]]
    ) -> int:
        """
        获取 context 部分的 token 数量。
        使用 tokenizer 本地计算，避免额外 API 调用。
        """
        context_str = tokenizer.apply_chat_template(
            context_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        # 使用 tokenizer 计算 token 数量
        context_tokens = tokenizer.encode(context_str, add_special_tokens=False)
        return len(context_tokens)

    def get_action_log_prob_sum(
        self, 
        client: openai.OpenAI,
        model: str,
        tokenizer: AutoTokenizer,
        special_tokens: set,
        context_messages: List[Dict[str, str]], 
        action_text: str
    ) -> Tuple[float, int]:
        """
        计算特定动作文本在给定上下文下的 LogProb 总和。
        
        Args:
            client: OpenAI 客户端（连接到 vLLM）
            model: 模型名称
            tokenizer: 对应的 tokenizer
            special_tokens: 需要过滤的特殊 token 集合
            context_messages: 上下文消息列表
            action_text: 需要计算 logprob 的动作文本
            
        Returns:
            (log_prob_sum, token_count): logprob 总和和有效 token 数量
        """
        # 1. 构建 Context 文本并计算 token 数量
        context_str = tokenizer.apply_chat_template(
            context_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        context_token_count = len(tokenizer.encode(context_str, add_special_tokens=False))
        
        # 2. 构建完整文本
        full_text = context_str + action_text
        
        # 3. 调用 vLLM API 获取 logprobs
        response = client.completions.create(
            model=model,
            prompt=full_text,
            max_tokens=0,       # 不生成新 token
            echo=True,          # 返回输入 token 的 logprob
            logprobs=1,         # 返回 top-1 logprob
            temperature=0
        )
        
        # 4. 提取 logprobs 数据
        logprobs_data = response.choices[0].logprobs
        tokens = logprobs_data.tokens           # token 文本列表
        token_logprobs = logprobs_data.token_logprobs  # 每个 token 的 logprob
        
        # 5. 计算 action 部分的 logprob 总和（跳过 context 和特殊 token）
        total_logprob = 0.0
        valid_token_count = 0
        
        for i in range(context_token_count, len(tokens)):
            token_text = tokens[i]
            logprob = token_logprobs[i]
            
            # 跳过特殊 token
            if token_text in special_tokens:
                continue
            
            # 跳过 None（通常是首 token）
            if logprob is None:
                continue
                
            total_logprob += logprob
            valid_token_count += 1
        
        return total_logprob, valid_token_count

    def calculate_disagreement(
        self, 
        context_messages: List[Dict[str, str]], 
        action_text: str
    ) -> Tuple[float, Dict]:
        """
        计算异构模型的归一化分歧分数。
        
        Args:
            context_messages: 上下文消息列表，如 [{"role": "user", "content": "..."}]
            action_text: 需要评估的动作文本（通常是 assistant 的回复）
            
        Returns:
            (normalized_score, details): 归一化分歧分数和详细信息
        """
        # 1. 计算物理常量：字节长度（作为归一化的公约数）
        byte_length = len(action_text.encode('utf-8'))
        
        # 极短文本平滑，防止除以极小数
        if byte_length < 2:
            return 0.0, {}
            
        # 2. 分别计算学生和教师模型的 LogProb 总和
        stu_log_sum, stu_cnt = self.get_action_log_prob_sum(
            self.stu_client, 
            self.stu_model, 
            self.stu_tokenizer,
            self.stu_special_tokens,
            context_messages, 
            action_text
        )
        
        tea_log_sum, tea_cnt = self.get_action_log_prob_sum(
            self.tea_client, 
            self.tea_model, 
            self.tea_tokenizer,
            self.tea_special_tokens,
            context_messages, 
            action_text
        )
        
        # 3. 计算差异
        # Diff = Student (High Prob, near 0) - Teacher (Low Prob, very negative)
        # Example: -2.0 - (-10.0) = 8.0 (High Disagreement)
        raw_diff = stu_log_sum - tea_log_sum
        
        # 4. 字节级归一化 (Bits per Byte Gap)
        normalized_score = raw_diff / byte_length
        
        details = {
            "stu_log_sum": stu_log_sum,
            "tea_log_sum": tea_log_sum,
            "stu_tokens": stu_cnt,
            "tea_tokens": tea_cnt,
            "byte_len": byte_length,
            "raw_diff": raw_diff
        }
        
        return normalized_score, details


# ================= 使用示例 =================

if __name__ == "__main__":
    # 示例：连接两个 vLLM 服务
    # 
    # 假设你已经启动了两个 vLLM 服务：
    # 
    # 学生模型 (Qwen2.5-7B):
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    #       --model /models/30B \
    #       --served_model_name student_model \
    #       --port 8100 --tensor_parallel_size 4 \
    #       --max_model_len 26384 \
    #     --gpu_memory_utilization 0.6
    #       --max_num_seqs 15

    # 注意：如果上面的路径不对，可以尝试：
    # --model TokerZ/7B-E2/actor
    # /models/30B/
    # CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    #     --model /model/32B \
    #     --served_model_name teacher_model \
    #     --port 8101 \
    #     --tensor_parallel_size 2
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    #     --model /model/32B \
    #     --served_model_name teacher_model \
    #     --port 8101 --tensor_parallel_size 4 \
    #     --max_model_len 26384 \
    #     --gpu_memory_utilization 0.5 \
    #     --max_num_seqs 16
    calculator = DisagreementCalculator(
        student_base_url="http://10.244.247.213:8100/v1",
        student_model="Qwen/Qwen2.5-7B-Instruct",
        student_tokenizer_path="Qwen/Qwen2.5-7B-Instruct",
        teacher_base_url="http://10.244.247.213:8101/v1",
        teacher_model="Qwen/Qwen3-30B-A3B",
        teacher_tokenizer_path="Qwen/Qwen3-30B-A3B",
    )

    # 测试用例
    context = [
        {"role": "user", "content": "Write a python function to add two numbers."}
    ]
    action = "def add(a, b):\n    return a + b"

    score, info = calculator.calculate_disagreement(context, action)

    print(f"Disagreement Score (BPB): {score:.4f}")
    print(f"Details: {info}")
