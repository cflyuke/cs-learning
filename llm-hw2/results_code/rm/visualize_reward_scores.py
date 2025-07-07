#!/usr/bin/env python3
"""
奖励函数得分可视化脚本
用于分析奖励模型在偏好数据集上的表现
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os

def load_reward_data(json_file):
    """Load reward score data"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_reward_distribution(data):
    """Analyze reward score distribution"""
    better_rewards = [d['better_reward'] for d in data]
    worse_rewards = [d['worse_reward'] for d in data]
    reward_diffs = [d['reward_difference'] for d in data]
    
    print("=== Reward Score Distribution Statistics ===")
    print(f"Total samples: {len(data)}")
    print(f"Better response average reward: {np.mean(better_rewards):.4f} ± {np.std(better_rewards):.4f}")
    print(f"Worse response average reward: {np.mean(worse_rewards):.4f} ± {np.std(worse_rewards):.4f}")
    print(f"Reward difference average: {np.mean(reward_diffs):.4f} ± {np.std(reward_diffs):.4f}")
    
    # Calculate accuracy
    correct_predictions = sum(1 for d in data if d['correct_prediction'])
    accuracy = correct_predictions / len(data)
    print(f"Model accuracy: {accuracy:.4f} ({correct_predictions}/{len(data)})")
    
    return better_rewards, worse_rewards, reward_diffs, accuracy

def plot_reward_distribution_comparison(data, output_dir="./pictures"):
    """Create reward score distribution comparison plot"""
    better_rewards, worse_rewards, reward_diffs, accuracy = analyze_reward_distribution(data)
    
    plt.figure(figsize=(10, 6))
    plt.hist(better_rewards, bins=30, alpha=0.7, label='Better Response', color='green', density=True)
    plt.hist(worse_rewards, bins=30, alpha=0.7, label='Worse Response', color='red', density=True)
    plt.xlabel('Reward Score')
    plt.ylabel('Density')
    plt.title(f'Reward Score Distribution Comparison (Accuracy: {accuracy:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'reward_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reward distribution comparison saved to: {output_path}")
    
    return better_rewards, worse_rewards, reward_diffs, accuracy

def plot_reward_difference_distribution(reward_diffs, accuracy, output_dir="./pictures"):
    """Create reward difference distribution plot"""
    plt.figure(figsize=(10, 6))
    plt.hist(reward_diffs, bins=30, alpha=0.7, color='blue', density=True)
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Difference Line')
    plt.xlabel('Reward Difference (Better - Worse)')
    plt.ylabel('Density')
    plt.title(f'Reward Difference Distribution (Accuracy: {accuracy:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'reward_difference_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reward difference distribution saved to: {output_path}")

def plot_reward_score_scatter(data, better_rewards, worse_rewards, accuracy, output_dir="./pictures"):
    """Create reward score scatter plot"""
    correct_mask = [d['correct_prediction'] for d in data]
    correct_better = [better_rewards[i] for i in range(len(data)) if correct_mask[i]]
    correct_worse = [worse_rewards[i] for i in range(len(data)) if correct_mask[i]]
    wrong_better = [better_rewards[i] for i in range(len(data)) if not correct_mask[i]]
    wrong_worse = [worse_rewards[i] for i in range(len(data)) if not correct_mask[i]]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(correct_worse, correct_better, alpha=0.6, color='green', 
               label=f'Correct Prediction ({len(correct_better)})', s=20)
    plt.scatter(wrong_worse, wrong_better, alpha=0.6, color='red', 
               label=f'Wrong Prediction ({len(wrong_better)})', s=20)
    
    # Add diagonal line
    min_val = min(min(better_rewards), min(worse_rewards))
    max_val = max(max(better_rewards), max(worse_rewards))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Line')
    
    plt.xlabel('Worse Response Reward')
    plt.ylabel('Better Response Reward')
    plt.title(f'Reward Score Scatter Plot (Accuracy: {accuracy:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'reward_score_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reward score scatter plot saved to: {output_path}")

def plot_prediction_accuracy_boxplot(data, reward_diffs, accuracy, output_dir="./pictures"):
    """Create prediction accuracy boxplot"""
    correct_mask = [d['correct_prediction'] for d in data]
    correct_diffs = [reward_diffs[i] for i in range(len(data)) if correct_mask[i]]
    wrong_diffs = [reward_diffs[i] for i in range(len(data)) if not correct_mask[i]]
    
    plt.figure(figsize=(8, 6))
    box_data = [correct_diffs, wrong_diffs]
    box_labels = ['Correct Prediction', 'Wrong Prediction']
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Difference Line')
    plt.ylabel('Reward Difference')
    plt.title(f'Prediction Accuracy vs Reward Difference (Accuracy: {accuracy:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'prediction_accuracy_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction accuracy boxplot saved to: {output_path}")

def create_all_visualizations(data, output_dir="./"):
    """Create all visualization plots as separate files"""
    print("\n=== Creating Visualization Plots ===")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Reward distribution comparison
    better_rewards, worse_rewards, reward_diffs, accuracy = plot_reward_distribution_comparison(data, output_dir)
    
    # 2. Reward difference distribution
    plot_reward_difference_distribution(reward_diffs, accuracy, output_dir)
    
    # 3. Reward score scatter plot
    plot_reward_score_scatter(data, better_rewards, worse_rewards, accuracy, output_dir)
    
    # 4. Prediction accuracy boxplot
    plot_prediction_accuracy_boxplot(data, reward_diffs, accuracy, output_dir)
    
    print(f"\nAll visualization plots saved to: {output_dir}")
    return accuracy

def analyze_error_cases(data, top_k=5):
    """Analyze error prediction cases"""
    print("\n=== Error Prediction Case Analysis ===")
    
    # Find error prediction cases
    error_cases = [d for d in data if not d['correct_prediction']]
    
    if not error_cases:
        print("No error prediction cases found!")
        return
    
    print(f"Total error predictions: {len(error_cases)}")
    
    # Sort by reward difference (largest absolute error)
    error_cases.sort(key=lambda x: abs(x['reward_difference']), reverse=True)
    
    print(f"\nTop {min(top_k, len(error_cases))} most severe error prediction cases:")
    for i, case in enumerate(error_cases[:top_k]):
        print(f"\n--- Error Case {i+1} ---")
        print(f"Reward difference: {case['reward_difference']:.4f}")
        print(f"Better response reward: {case['better_reward']:.4f}")
        print(f"Worse response reward: {case['worse_reward']:.4f}")
        print(f"Better response: {case['better_text'][:200]}...")
        print(f"Worse response: {case['worse_text'][:200]}...")

def main():
    parser = argparse.ArgumentParser(description='Reward Model Visualization Analysis')
    parser.add_argument('--json_file', type=str, 
                       default='outputs/qwen_2_5_rm/reward_scores_visualization.json',
                       help='Reward scores JSON file path')
    parser.add_argument('--output_dir', type=str, default='outputs/qwen_2_5_rm/pictures',
                       help='Output directory')
    parser.add_argument('--show_errors', action='store_true',
                       help='Show error prediction cases')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File not found {args.json_file}")
        print("Please ensure you have run the reward model training and generated the visualization data file")
        return
    
    # Load data
    print(f"Loading data: {args.json_file}")
    data = load_reward_data(args.json_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create visualizations
    accuracy = create_all_visualizations(data, args.output_dir)
    
    # Analyze error cases
    if args.show_errors:
        analyze_error_cases(data)
    
    print(f"\nAnalysis completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
