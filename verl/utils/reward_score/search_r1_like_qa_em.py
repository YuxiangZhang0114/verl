# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
import os
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# LLM matching configuration
LLM_MATCH_BASE_URL = os.environ.get("LLM_MATCH_BASE_URL", "http://localhost:9527/v1")
LLM_MATCH_MODEL = os.environ.get("LLM_MATCH_MODEL", "evaluator")
LLM_MATCH_ENABLED = os.environ.get("LLM_MATCH_ENABLED", "true").lower() == "true"

# Global OpenAI client for LLM matching
_llm_client = None


def _get_llm_client():
    """Get or create the OpenAI client for LLM matching."""
    global _llm_client
    if _llm_client is None and OpenAI is not None:
        _llm_client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require a real API key
            base_url=LLM_MATCH_BASE_URL,
        )
    return _llm_client


def llm_match(prediction: str, golden_answer: str) -> int:
    """
    Use LLM to determine if the prediction semantically matches the golden answer.
    
    Args:
        prediction: The predicted answer string
        golden_answer: The golden answer string
        
    Returns:
        1 if the LLM determines they match, 0 otherwise
    """
    if not LLM_MATCH_ENABLED or OpenAI is None:
        return 0
    
    client = _get_llm_client()
    if client is None:
        return 0
    
    # Strict and rigorous prompt for answer matching
    prompt = f"""You are an expert evaluator tasked with determining whether two answers are semantically equivalent and correct.

CRITICAL EVALUATION CRITERIA:
1. The answers must convey the SAME core information, facts, or meaning
2. Minor differences in wording, formatting, or phrasing are acceptable ONLY if the semantic content is identical
3. Numerical values must match exactly (e.g., "5" vs "five" is acceptable, but "5" vs "6" is not)
4. Proper nouns, names, and specific entities must match exactly
5. The answer must be factually correct and complete - partial or incorrect information does not match
6. Synonyms are acceptable ONLY if they preserve the exact meaning without ambiguity
7. Different units of measurement are NOT equivalent unless explicitly converted (e.g., "5 meters" vs "500 centimeters" is acceptable, but "5 meters" vs "5 feet" is not)
8. Generic words like "answer", "solution", or placeholder text are NOT valid answers and should NEVER match

EVALUATION INSTRUCTIONS:
- Be STRICT and RIGOROUS in your assessment
- Do NOT accept matches if there is any ambiguity, incompleteness, or factual discrepancy
- REJECT any prediction that is just a generic word like "answer", "solution", or similar placeholder text
- Consider context: if the question requires a specific format or precision, both answers must meet that requirement
- When in DOUBT or UNCERTAIN, you MUST return "NO" - only return "YES" if you are CERTAIN the answers match
- Return ONLY "YES" if you are CERTAIN the answers match, or "NO" if they do not match or if you are uncertain
- Do not provide explanations, justifications, or additional text - only "YES" or "NO"

PREDICTED ANSWER: 
{prediction}


GOLDEN ANSWER: 
{golden_answer}

Do these answers match with each other? Respond with only "YES" or "NO":"""
    # print(prompt)
    try:
        response = client.chat.completions.create(
            model=LLM_MATCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise and rigorous answer matching evaluator. You must be strict and conservative. When in doubt or uncertain, you MUST return 'NO'. Only return 'YES' if you are certain the answers match. Only return 'YES' or 'NO'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Use deterministic output
            max_tokens=15,  # Only need YES/NO
        )
        
        result = response.choices[0].message.content.strip().upper()
        # Be conservative: only accept if result clearly starts with "YES" and contains no "NO"
        # This ensures we only match when the LLM is certain
        if result.startswith("YES") and "NO" not in result:
            return 1
        else:
            return 0
    except Exception as e:
        # If LLM matching fails, return 0 (no match)

        print(f"Warning: LLM matching failed: {e}")
        return 0


def em_check(prediction, golden_answers):
    """
    Check if prediction exactly matches any golden answer.
    If exact match fails and LLM matching is enabled, use LLM to check semantic equivalence.
    
    Args:
        prediction: The predicted answer string
        golden_answers: A string or list of golden answer strings
        
    Returns:
        1 if match found (exact or LLM), 0 otherwise
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    
    # First try exact match
    for golden_answer in golden_answers:
        golden_answer_normalized = normalize_answer(golden_answer)
        if golden_answer_normalized == normalized_prediction:
            score = 1
            break
    
    # If exact match failed and LLM matching is enabled, try LLM matching
    if score == 0 and LLM_MATCH_ENABLED:
        for golden_answer in golden_answers:
            if llm_match(prediction, golden_answer):
                score = 1
                break
    
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction and len(normalized_prediction) < len(golden_answer)*2:
            score = 1
            break
    return score


    return opening_tags,
def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags

def count_thinking_tags(text):
    opening_tags = text.count("<thinking>")
    closing_tags = text.count("</thinking>")

    return opening_tags, closing_tags

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 128) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_asearcher_with_thinking(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 100) == 1
    if do_print:
        print("--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")

    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
