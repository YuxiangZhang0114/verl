"""
Compare disagreement score distributions between correct and incorrect answer samples
Only analyze samples with more than 2 turns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

# Set plot style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)


import re
def extract_answer(content):
    # <answer>...</answer>
    answer = re.search(r"<answer>(.*?)</answer>", content)
    if answer:
        return answer.group(1)
    else:
        return None

import string
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


def em_check(prediction, golden_answer):
    """
    Check if prediction exactly matches golden answer.
    """
    normalized_prediction = normalize_answer(str(prediction))
    normalized_golden_answer = normalize_answer(str(golden_answer))
    return normalized_prediction == normalized_golden_answer

def load_disagreement_scores(jsonl_file, filter_correct=True, min_turns=2):
    """
    Load JSONL file and extract disagreement scores
    
    Args:
        jsonl_file: Path to JSONL file
        filter_correct: True = keep correct answer samples, False = keep incorrect answer samples
        min_turns: Minimum number of turns to include (default: 2, only analyze samples with >2 turns)
    """
    
    all_scores = []
    sample_max_scores = []
    sample_avg_scores = []
    turn_idx_scores = defaultdict(list)
    sample_info = []
    
    filtered_count = 0
    total_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # Filter by answer correctness
                # answer_in_content = data['answer'].lower() in data['turn_logprobs'][-1]['raw_content'].lower()
                
                answer_in_content = em_check(data['answer'], extract_answer(data['turn_logprobs'][-1]['raw_content']))
                if filter_correct:
                    # Keep only correct answer samples
                    if not answer_in_content:
                        continue
                else:
                    # Keep only incorrect answer samples
                    if answer_in_content:
                        continue
                
                turn_logprobs = data.get("turn_logprobs", [])
                
                sample_scores = []
                for turn in turn_logprobs:
                    if "disagreement_score" in turn and "error" not in turn:
                        score = turn["disagreement_score"]
                        all_scores.append(score)
                        sample_scores.append(score)
                        
                        turn_idx = turn.get("turn_idx", -1)
                        turn_idx_scores[turn_idx].append(score)
                
                # Filter by number of turns (only keep samples with > min_turns)
                if sample_scores and len(sample_scores) > min_turns:
                    sample_max_scores.append(max(sample_scores))
                    sample_avg_scores.append(np.mean(sample_scores))
                    sample_info.append({
                        'index': data.get('index', -1),
                        'max_score': max(sample_scores),
                        'avg_score': np.mean(sample_scores),
                        'num_turns': len(sample_scores)
                    })
                elif sample_scores:
                    filtered_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"⚠ Error parsing line {line_num}: {e}")
                continue
    
    print(f"  Filtered out {filtered_count} samples with <= {min_turns} turns")
    
    return {
        'all_scores': np.array(all_scores),
        'sample_max_scores': np.array(sample_max_scores),
        'sample_avg_scores': np.array(sample_avg_scores),
        'turn_idx_scores': turn_idx_scores,
        'sample_info': sample_info
    }


def plot_comparison(correct_data, incorrect_data, save_path='distribution_comparison.png'):
    """
    Compare two distributions and generate visualizations
    """
    
    # correct_scores = correct_data['sample_max_scores']
    # incorrect_scores = incorrect_data['sample_max_scores']
    correct_scores = correct_data['all_scores']
    incorrect_scores = incorrect_data['all_scores']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overlapping Histogram Comparison
    bins = np.linspace(0, 0.35, 50)
    axes[0, 0].hist(correct_scores, bins=bins, alpha=0.6, label='Correct Answer', 
                    color='green', density=True, edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(incorrect_scores, bins=bins, alpha=0.6, label='Incorrect Answer', 
                    color='red', density=True, edgecolor='black', linewidth=0.5)
    axes[0, 0].axvline(np.mean(correct_scores), color='darkgreen', linestyle='--', 
                       linewidth=2, label=f'Correct Mean: {np.mean(correct_scores):.3f}')
    axes[0, 0].axvline(np.mean(incorrect_scores), color='darkred', linestyle='--', 
                       linewidth=2, label=f'Incorrect Mean: {np.mean(incorrect_scores):.3f}')
    axes[0, 0].set_xlabel('Max Disagreement Score per Sample', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Distribution Comparison of Max Disagreement Score (Overlapping Histogram)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10, loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 0.35)
    
    # 2. Kernel Density Estimation (KDE) Comparison
    from scipy.stats import gaussian_kde
    
    kde_correct = gaussian_kde(correct_scores)
    kde_incorrect = gaussian_kde(incorrect_scores)
    x_range = np.linspace(0, 0.35, 500)
    
    axes[0, 1].plot(x_range, kde_correct(x_range), linewidth=2.5, 
                    label=f'Correct Answer (n={len(correct_scores)})', color='green')
    axes[0, 1].plot(x_range, kde_incorrect(x_range), linewidth=2.5, 
                    label=f'Incorrect Answer (n={len(incorrect_scores)})', color='red')
    axes[0, 1].fill_between(x_range, kde_correct(x_range), alpha=0.3, color='green')
    axes[0, 1].fill_between(x_range, kde_incorrect(x_range), alpha=0.3, color='red')
    axes[0, 1].axvline(np.mean(correct_scores), color='darkgreen', 
                       linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].axvline(np.mean(incorrect_scores), color='darkred', 
                       linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Max Disagreement Score per Sample', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Kernel Density Estimation Comparison (KDE)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 0.35)
    
    # 3. Cumulative Distribution Function (CDF) Comparison
    sorted_correct = np.sort(correct_scores)
    sorted_incorrect = np.sort(incorrect_scores)
    cdf_correct = np.arange(1, len(sorted_correct) + 1) / len(sorted_correct)
    cdf_incorrect = np.arange(1, len(sorted_incorrect) + 1) / len(sorted_incorrect)
    
    axes[1, 0].plot(sorted_correct, cdf_correct, linewidth=2.5, 
                    label='Correct Answer', color='green')
    axes[1, 0].plot(sorted_incorrect, cdf_incorrect, linewidth=2.5, 
                    label='Incorrect Answer', color='red')
    axes[1, 0].axvline(0.2, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    axes[1, 0].axvline(0.3, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    axes[1, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Max Disagreement Score per Sample', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1, 0].set_title('Cumulative Distribution Function Comparison (CDF)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 0.35)
    
    # 4. Box Plot + Violin Plot Comparison
    bp = axes[1, 1].boxplot([correct_scores, incorrect_scores], 
                             labels=['Correct Answer', 'Incorrect Answer'],
                             patch_artist=True,
                             widths=0.6,
                             showmeans=True,
                             meanline=True)
    
    # Set box plot colors
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add violin plot overlay
    parts = axes[1, 1].violinplot([correct_scores, incorrect_scores], 
                                   positions=[1, 2], 
                                   widths=0.8,
                                   showmeans=False, 
                                   showmedians=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    axes[1, 1].set_ylabel('Max Disagreement Score per Sample', fontsize=12)
    axes[1, 1].set_title('Box Plot + Violin Plot Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 0.35)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Plot saved: {save_path}")
    plt.show()
    
    # Print statistical comparison
    print("\n" + "=" * 80)
    print("Statistical Comparison Analysis")
    print("=" * 80)
    
    print(f"\n【Correct Answer】 Sample count: {len(correct_scores)}")
    print(f"  Mean:     {np.mean(correct_scores):.4f}")
    print(f"  Median:   {np.median(correct_scores):.4f}")
    print(f"  Std Dev:  {np.std(correct_scores):.4f}")
    print(f"  Min:      {np.min(correct_scores):.4f}")
    print(f"  Max:      {np.max(correct_scores):.4f}")
    
    print(f"\n【Incorrect Answer】 Sample count: {len(incorrect_scores)}")
    print(f"  Mean:     {np.mean(incorrect_scores):.4f}")
    print(f"  Median:   {np.median(incorrect_scores):.4f}")
    print(f"  Std Dev:  {np.std(incorrect_scores):.4f}")
    print(f"  Min:      {np.min(incorrect_scores):.4f}")
    print(f"  Max:      {np.max(incorrect_scores):.4f}")
    
    # Statistical tests
    print("\n" + "=" * 80)
    print("Statistical Tests")
    print("=" * 80)
    
    # t-test
    t_stat, t_pval = stats.ttest_ind(correct_scores, incorrect_scores)
    print(f"\nT-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {t_pval:.6f}")
    print(f"  Conclusion:  {'Significant difference' if t_pval < 0.05 else 'No significant difference'} (α=0.05)")
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(correct_scores, incorrect_scores, alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {u_stat:.4f}")
    print(f"  p-value:     {u_pval:.6f}")
    print(f"  Conclusion:  {'Significant difference' if u_pval < 0.05 else 'No significant difference'} (α=0.05)")
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(correct_scores)-1)*np.var(correct_scores, ddof=1) + 
                          (len(incorrect_scores)-1)*np.var(incorrect_scores, ddof=1)) / 
                         (len(correct_scores) + len(incorrect_scores) - 2))
    cohens_d = (np.mean(correct_scores) - np.mean(incorrect_scores)) / pooled_std
    print(f"\nCohen's d (effect size):")
    print(f"  d-value:     {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        effect_size = "very small"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"  Interpretation: {effect_size} effect")
    
    # Threshold analysis
    print("\n" + "=" * 80)
    print("Threshold Analysis Comparison")
    print("=" * 80)
    
    for threshold in [0.15, 0.2, 0.25, 0.3, 0.35]:
        correct_count = (correct_scores > threshold).sum()
        correct_ratio = correct_count / len(correct_scores) * 100
        incorrect_count = (incorrect_scores > threshold).sum()
        incorrect_ratio = incorrect_count / len(incorrect_scores) * 100
        
        print(f"\nThreshold > {threshold}:")
        print(f"  Correct:   {correct_count:4d} ({correct_ratio:5.1f}%)")
        print(f"  Incorrect: {incorrect_count:4d} ({incorrect_ratio:5.1f}%)")
        print(f"  Difference: {incorrect_ratio - correct_ratio:+5.1f}%")


if __name__ == "__main__":
    # Load data
    jsonl_file = "data/train_student_42step_32B_teacher_logprobs.jsonl"
    
    print("Loading correct answer samples...")
    correct_data = load_disagreement_scores(jsonl_file, filter_correct=True, min_turns=1)
    print(f"✓ Loaded: {len(correct_data['sample_max_scores'])} samples")
    
    print("\nLoading incorrect answer samples...")
    incorrect_data = load_disagreement_scores(jsonl_file, filter_correct=False, min_turns=1)
    print(f"✓ Loaded: {len(incorrect_data['sample_max_scores'])} samples")
    
    # Generate comparison plots
    plot_comparison(correct_data, incorrect_data)
