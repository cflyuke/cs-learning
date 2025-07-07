#!/usr/bin/env python3
"""
DPO模型分析脚本
基于align_anything_t2t数据集分析DPO微调对模型行为的影响
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
from results.dpo.custom_reward_model import load_custom_reward_model, calculate_reward_score_with_custom_model
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DPOAnalyzer:
    def __init__(self, 
                 original_model_path="./Qwen2.5-0.5B-Instruct",
                 dpo_model_path="./qwen_2_5_dpo/slice_end",
                 reward_model_path="./qwen_2_5_rm/slice_end",
                 device="cpu"):
        """
        初始化DPO分析器
        """
        self.device = device
        self.original_model_path = original_model_path
        self.dpo_model_path = dpo_model_path
        self.reward_model_path = reward_model_path
        
        print("正在加载模型...")
        self.load_models()
        
    def load_models(self):
        """加载所有需要的模型"""
        try:
            # 加载原始模型
            print("加载原始模型...")
            self.original_model = AutoModelForCausalLM.from_pretrained(
                self.original_model_path, 
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ).to(self.device)
            self.original_tokenizer = AutoTokenizer.from_pretrained(self.original_model_path)
            
            # 加载DPO微调模型
            print("加载DPO微调模型...")
            self.dpo_model = AutoModelForCausalLM.from_pretrained(
                self.dpo_model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ).to(self.device)
            self.dpo_tokenizer = AutoTokenizer.from_pretrained(self.dpo_model_path)
            
            # 尝试加载奖励模型
            try:
                torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
                self.reward_model, self.reward_tokenizer = load_custom_reward_model(
                    self.reward_model_path,
                    device=self.device,
                    torch_dtype=torch_dtype
                )
                self.has_reward_model = True
                print("奖励模型加载成功！")
            except Exception as e:
                print(f"无法加载奖励模型: {e}")
                print("将跳过奖励模型评分部分")
                self.has_reward_model = False
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def generate_response(self, model, tokenizer, question, max_new_tokens=512):
        """使用指定模型生成回复"""
        messages = [
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def calculate_reward_score(self, question, response):
        """计算奖励分数（如果有奖励模型）"""
        if not self.has_reward_model:
            return None
            
        try:
            # 使用自定义的奖励分数计算函数
            reward_score = calculate_reward_score_with_custom_model(
                self.reward_model, 
                self.reward_tokenizer, 
                question, 
                response, 
                self.device
            )
            return reward_score
        except Exception as e:
            print(f"计算奖励分数时出错: {e}")
            return None
    
    def load_test_data(self, data_path="./align_anything_t2t/val_1k.parquet", sample_size=50):
        """加载测试数据"""
        print(f"加载测试数据: {data_path}")
        df = pd.read_parquet(data_path)
        
        # 随机采样以减少计算时间
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"数据集大小: {len(df)}")
        print(f"数据集列: {list(df.columns)}")
        
        return df
    
    def analyze_responses(self, df, output_dir="./analysis_results"):
        """分析模型回复"""
        Path(output_dir).mkdir(exist_ok=True)
        
        results = []
        
        print("开始生成回复并分析...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            
            # 生成原始模型回复
            original_response = self.generate_response(self.original_model, self.original_tokenizer, question)
            
            # 生成DPO模型回复
            dpo_response = self.generate_response(self.dpo_model, self.dpo_tokenizer, question)
            
            # 计算奖励分数
            original_reward = self.calculate_reward_score(question, original_response)
            dpo_reward = self.calculate_reward_score(question, dpo_response)
            
            result = {
                'question': question,
                'original_response': original_response,
                'dpo_response': dpo_response,
                'original_reward': original_reward,
                'dpo_reward': dpo_reward,
                'reward_diff': dpo_reward - original_reward if (dpo_reward is not None and original_reward is not None) else None,
            }
            
            results.append(result)
        
        # 保存结果
        results_file = Path(output_dir) / "analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"分析结果已保存到: {results_file}")
        return results
    
    def create_visualizations(self, results, output_dir="./analysis_results"):
        """创建可视化图表"""
        print("创建可视化图表...")
        
        # 过滤有效的奖励分数
        valid_results = [r for r in results if r['reward_diff'] is not None]
        
        if not valid_results:
            print("没有有效的奖励分数数据，跳过奖励相关的可视化")
            return
        
        reward_diffs = [r['reward_diff'] for r in valid_results]
        original_rewards = [r['original_reward'] for r in valid_results]
        dpo_rewards = [r['dpo_reward'] for r in valid_results]
        
        # 1. 奖励分数分布对比
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(original_rewards, bins=20, alpha=0.7, label='原始模型', color='blue')
        plt.hist(dpo_rewards, bins=20, alpha=0.7, label='DPO模型', color='red')
        plt.xlabel('奖励分数')
        plt.ylabel('频次')
        plt.title('奖励分数分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 奖励分数差异分布
        plt.subplot(2, 2, 2)
        plt.hist(reward_diffs, bins=20, alpha=0.7, color='green')
        plt.axvline(x=0, color='red', linestyle='--', label='零差异线')
        plt.xlabel('奖励分数差异 (DPO - 原始)')
        plt.ylabel('频次')
        plt.title('奖励分数差异分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 散点图
        plt.subplot(2, 2, 3)
        plt.scatter(original_rewards, dpo_rewards, alpha=0.6)
        min_val = min(min(original_rewards), min(dpo_rewards))
        max_val = max(max(original_rewards), max(dpo_rewards))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        plt.xlabel('原始模型奖励分数')
        plt.ylabel('DPO模型奖励分数')
        plt.title('奖励分数散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 统计信息
        plt.subplot(2, 2, 4)
        stats_text = f"""
        统计信息:
        
        原始模型平均奖励: {np.mean(original_rewards):.4f}
        DPO模型平均奖励: {np.mean(dpo_rewards):.4f}
        平均奖励差异: {np.mean(reward_diffs):.4f}
        
        DPO模型更好的案例: {sum(1 for d in reward_diffs if d > 0)} / {len(reward_diffs)}
        改进比例: {sum(1 for d in reward_diffs if d > 0) / len(reward_diffs) * 100:.1f}%
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "reward_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {Path(output_dir) / 'reward_analysis.png'}")
    
    def analyze_cases(self, results, output_dir="./analysis_results", top_k=30):
        """分析具体案例"""
        print("分析具体案例...")
        
        # 过滤有效结果
        valid_results = [r for r in results if r['reward_diff'] is not None]
        
        if not valid_results:
            print("没有有效的奖励分数数据")
            return
        
        # 按奖励差异排序
        valid_results.sort(key=lambda x: x['reward_diff'], reverse=True)
        
        # 分析最佳改进案例
        best_cases = valid_results[:top_k]
        worst_cases = valid_results[-top_k:]
        
        analysis_text = "# DPO模型分析报告\n\n"
        
        # 总体统计
        reward_diffs = [r['reward_diff'] for r in valid_results]
        improved_count = sum(1 for d in reward_diffs if d > 0)
        total_count = len(reward_diffs)
        
        analysis_text += f"## 总体统计\n"
        analysis_text += f"- 总测试案例数: {total_count}\n"
        analysis_text += f"- DPO模型表现更好的案例: {improved_count} ({improved_count/total_count*100:.1f}%)\n"
        analysis_text += f"- 平均奖励分数提升: {np.mean(reward_diffs):.4f}\n"
        analysis_text += f"- 奖励分数提升标准差: {np.std(reward_diffs):.4f}\n\n"
        
        # 最佳改进案例
        analysis_text += f"## 最佳改进案例 (Top {top_k})\n\n"
        for i, case in enumerate(best_cases):
            analysis_text += f"### 案例 {i+1} (奖励提升: {case['reward_diff']:.4f})\n"
            analysis_text += f"**问题**: {case['question'][:200]}...\n\n"
            analysis_text += f"**原始模型回复**: {case['original_response'][:300]}...\n\n"
            analysis_text += f"**DPO模型回复**: {case['dpo_response'][:300]}...\n\n"
            analysis_text += "---\n\n"
        
        # 最差案例
        analysis_text += f"## 表现下降案例 (Bottom {top_k})\n\n"
        for i, case in enumerate(worst_cases):
            analysis_text += f"### 案例 {i+1} (奖励下降: {case['reward_diff']:.4f})\n"
            analysis_text += f"**问题**: {case['question'][:200]}...\n\n"
            analysis_text += f"**原始模型回复**: {case['original_response'][:300]}...\n\n"
            analysis_text += f"**DPO模型回复**: {case['dpo_response'][:300]}...\n\n"
            analysis_text += "---\n\n"
        
        # 保存分析报告
        report_file = Path(output_dir) / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        
        print(f"分析报告已保存到: {report_file}")
    
    def run_full_analysis(self, data_path="./align_anything_t2t/val_1k.parquet", 
                         sample_size=50, output_dir="./analysis_results"):
        """运行完整分析"""
        print("开始DPO模型完整分析...")
        
        # 加载数据
        df = self.load_test_data(data_path, sample_size)
        
        # 分析回复
        results = self.analyze_responses(df, output_dir)
        
        # 创建可视化
        self.create_visualizations(results, output_dir)
        
        # 分析具体案例
        self.analyze_cases(results, output_dir)
        
        print("分析完成！")
        return results

def main():
    parser = argparse.ArgumentParser(description='DPO模型分析脚本')
    parser.add_argument('--original_model', type=str, default="./Qwen2.5-0.5B-Instruct",
                       help='原始模型路径')
    parser.add_argument('--dpo_model', type=str, default="./qwen_2_5_dpo/slice_end",
                       help='DPO微调模型路径')
    parser.add_argument('--reward_model', type=str, default="./qwen_2_5_rm/slice_end",
                       help='奖励模型路径')
    parser.add_argument('--data_path', type=str, default="./align_anything_t2t/val_1k.parquet",
                       help='测试数据路径')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='采样大小')
    parser.add_argument('--output_dir', type=str, default="./analysis_results",
                       help='输出目录')
    parser.add_argument('--device', type=str, default="cpu",
                       help='设备类型 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DPOAnalyzer(
        original_model_path=args.original_model,
        dpo_model_path=args.dpo_model,
        reward_model_path=args.reward_model,
        device=args.device
    )
    
    # 运行分析
    results = analyzer.run_full_analysis(
        data_path=args.data_path,
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )
    
    print("DPO分析完成！")

if __name__ == "__main__":
    main()
