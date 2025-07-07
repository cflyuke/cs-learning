"""专门为LAG环境设计的模型渲染脚本"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "HARL"))

from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner


def parse_args():
    parser = argparse.ArgumentParser(description='LAG环境模型渲染演示')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='训练模型目录路径')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    parser.add_argument('--render', action='store_true', default=True,
                        help='是否显示渲染')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='步骤间延迟时间(秒)')
    return parser.parse_args()


def load_config(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_runner(config, model_dir, seed):
    """创建Runner"""
    main_args = config['main_args']
    env_args = config['env_args']
    algo_args = config['algo_args'].copy()
    
    algo_args['seed'] = {'seed': seed, 'seed_specify': True}
    algo_args['render'] = {'use_render': True, 'render_episodes': 1}
    algo_args['train'] = algo_args.get('train', {})
    algo_args['train']['model_dir'] = None
    
    algo = main_args['algo']
    
    if algo in ['happo', 'happo_optimized']:
        # on-policy算法
        runner = OnPolicyHARunner(main_args, algo_args, env_args)
    elif algo == 'hasac':
        # off-policy算法
        runner = OffPolicyHARunner(main_args, algo_args, env_args)
    else:
        raise ValueError(f"不支持的算法: {algo}")
    
    models_dir = os.path.join(model_dir, 'models')
    runner.algo_args['train']['model_dir'] = models_dir
    
    return runner


def main():
    args = parse_args()
    
    print(f"LAG环境模型渲染演示")
    print(f"模型目录: {args.model_dir}")
    print(f"随机种子: {args.seed}")
    print("-" * 50)
    
    try:
        config = load_config(args.model_dir)
        print(f"实验名称: {config['main_args']['exp_name']}")
        print(f"算法: {config['main_args']['algo']}")
        print(f"环境: {config['main_args']['env']}")
        print(f"场景: {config['env_args']['scenario']}")
        print(f"任务: {config['env_args']['task']}")
        
        print("\n创建Runner和环境...")
        runner = create_runner(config, args.model_dir, args.seed)
        print("✓ Runner和环境创建成功")
        
        print("加载训练模型...")
        runner.restore()
        print("✓ 模型加载成功")

        print("\n开始渲染...")
        runner.render()
        print("✓ 渲染完成")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            runner.close()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())
