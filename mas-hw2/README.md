# MAS-HW2: Multi-Agent Reinforcement Learning with Optimized HAPPO

本项目实现了基于HARL框架的多智能体强化学习算法，主要包含优化版HAPPO算法的实现和LAG环境下的训练与渲染。

## 目录结构

### 整体目录说明
```
mas-hw2/
├── README.md                    # 项目说明文档
├── report.pdf                  # 实验报告
├── code/                       # 核心代码目录
│   ├── happo_optimized.py      # 优化版HAPPO算法实现
│   ├── happo_optimized.yaml    # HAPPO优化算法配置文件
│   └── lag_render.py           # LAG环境模型渲染脚本
└── results/                    # 实验结果目录
   ├── dump_result/            # 基础实验结果
   ├── happo/                  # 标准HAPPO实验结果
   ├── happo_optimized/        # 优化HAPPO实验结果
   ├── happo_optimized_fine_tune/ # 优化HAPPO微调结果
   ├── hasac/                  # HASAC算法实验结果
   ├── pictures/               # 实验图表和可视化结果
   └── render/                 # 渲染输出文件

```
### 训练结果目录说明

- **dump_result/**: 基础实验的训练日志和模型文件
- **happo/**: 标准HAPPO算法的完整训练结果
  - `config.json`: 训练配置
  - `progress.txt`: 训练进度日志
  - `logs/`: TensorBoard日志文件
  - `models/`: 训练好的模型文件
- **happo_optimized/**: 优化版HAPPO算法训练结果
- **happo_optimized_fine_tune/**: 优化算法的微调结果
- **hasac/**: HASAC算法对比实验结果
- **pictures/**: 包含各种实验图表
  - `happo_optimized.png`: 优化算法性能曲线
  - `happo_hasac.png`: 算法对比图
  - `agent0_dist_entropy.png`: 智能体0的分布熵变化
  - `agent1_dist_entropy.png`: 智能体1的分布熵变化
- **render/**: 渲染输出文件
  - `*.acmi`: 各算法的渲染记录文件
  - `bash.log`: 渲染过程日志


## 使用方法

### 环境准备
配置好harl库所需和lag环境即可

### 模型检验和渲染(lag_render.py)
`lag_render.py` 是专门为LAG环境设计的模型渲染脚本，用于检验和可视化训练好的智能体策略。

#### 基本用法
```bash
python code/lag_render.py --model_dir results/happo_optimized
```

#### 参数说明
- `--model_dir`: **必需参数**，指定训练模型的目录路径
- `--seed`: 随机种子，默认为1
- `--render`: 是否显示渲染，默认为True
- `--delay`: 步骤间延迟时间(秒)，默认为0.1

#### 使用示例
```bash
# 渲染优化版HAPPO模型
python code/lag_render.py --model_dir results/happo_optimized --seed 42 --delay 0.05

# 渲染标准HAPPO模型
python code/lag_render.py --model_dir results/happo --seed 1

# 渲染HASAC模型
python code/lag_render.py --model_dir results/hasac
```

### 训练模型
1. 将`code/happo_optimized.py`放入`HARL/harl/algorithms/actors`文件夹
2. 将`code/happo_optimized.yaml`放入`HARL/harl/configs/algos_cfgs`文件夹
3. 在`HARL/examples/train.py`，`HARL/harl/algorithms/actors/__init__.py`, `HARL/harl/runners/__init__.py`中注册`happo_optimized`

```bash
# 使用优化版HAPPO训练
cd HARL
python examples/train.py --algo happo_optimized --env lag --exp_name happo_optimized_lag_2v2_noweapon_vsbaseline  --scenario MultipleCombat --task "2v2/NoWeapon/vsBaseline" 
```


