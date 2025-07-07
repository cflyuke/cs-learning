# 多智能体强化学习-供应链管理系统

本项目实现了一个基于多智能体强化学习的供应链管理系统，用于模拟和优化多级供应链中的库存管理决策。

## 项目结构

- `main.py`：主程序，包含配置、训练、测试和可视化功能，如果要对实验进行复现，调整配置config和主函数部分即可
- `env.py`：环境模拟，实现供应链动态和奖励计算
- `agents.py`：各种智能体的实现，包括Random、DQN、BS和MOE智能体
- `models_trained/`：预训练模型存储目录
  - `reshape_reward/`：使用奖励重塑训练的模型
  - `noreshape_reward/`：不使用奖励重塑训练的模型
  - `bs_firm_1_final.pth`：在三个企业，另外两个企业采用BS策略实验训练的模型
- `report.pdf`：项目报告

## 使用方法
在`main.py`中有说明，可以进行模型训练和测试，也可以利用训练好的模型进行实验复现，每次实验的配置在`main.py`中的config类进行设置。

## 依赖库
- numpy
- matplotlib
- torch (PyTorch)
- collections
- os
- random
