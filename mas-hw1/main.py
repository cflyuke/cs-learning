import numpy as np
import matplotlib.pyplot as plt
import os
from env import Env
from agents import DQNAgent, BSAgent, RandomAgent, MOEAgent


class Config:
    def __init__(self):
        self.max_step = 100 # env的最大游戏步长
        self.eps_start = 1.0 # 探索机制开始值
        self.eps_end = 0.01 # 探索概率结束值
        self.eps_decay = 0.995 # 探索衰减值

        self.state_size = 5 # 状态空间大小
        self.action_size = 20 # 动作空间大小
        self.max_order = self.action_size # 最大订单数
        self.buffer_size = 10000 # 回放缓冲区大小
        self.batch_size = 64 # 批大小
        self.gamma = 0.99 # 折扣因子
        self.learning_rate = 1e-3 # 学习率
        self.tau = 1e-3 # 软更新参数
        self.update_every = 3 # 更新目标网络的频率

        self.num_firms = 4 # 公司数量
        self.beta = np.array([[0.3], [0.3], [0.3], [0.3]]) # 奖励重塑系数
        self.initial_inventory = [20, 100, 100, 100]  # 初始库存
        self.p = [10, 9, 8, 7]  # 价格列表
        self.h = 0.5  # 库存持有成本
        self.c = 2  # 损失销售成本
        self.poisson_lambda = 10  # 泊松分布的均值
        self.max_steps = 100  # 每个episode的最大步数

        
def train_dqn(env, agents, num_episodes, config, use_reshape_reward=False):
    """
    训练DQN智能体
    
    :param env: 环境
    :param agents: 智能体列表(包含DQNAgent和BSAgent)
    :param num_episodes: 训练的episodes数量
    :param config: 配置信息
    :param use_reshape_reward: 是否使用重塑奖励来训练
    :return: 每个DQNAgent的所有episode奖励
    """
    # 筛选出DQNAgent并初始化分数记录
    dqn_agents = [a for a in agents if isinstance(a, DQNAgent)]
    scores = [[] for _ in agents]
    eps = config.eps_start  # 初始epsilon值
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        episode_scores = np.zeros(len(agents))
        
        for t in range(config.max_steps):
            # 所有企业采取动作
            actions = np.zeros((env.num_firms, 1))
            for agent in agents:
                firm_state = state[agent.firm_id].reshape(1, -1)
                if isinstance(agent, DQNAgent):
                    action = agent.act(firm_state, eps)
                else:
                    action = agent.act(firm_state)
                actions[agent.firm_id] = action
            
            # 执行动作
            next_state, rewards, done, reshape_rewards = env.step(actions)
            
            # 对Agent进行学习和记录分数
            for agent in agents:
                reward = rewards[agent.firm_id][0]
                reshape_reward = reshape_rewards[agent.firm_id][0]
                episode_scores[agent.firm_id] += reward
                if agent in dqn_agents:
                    if use_reshape_reward:
                        agent.step(state[agent.firm_id].reshape(1, -1),
                                actions[agent.firm_id],
                                reshape_reward,
                                next_state[agent.firm_id].reshape(1, -1),
                                done)
                    else:
                        agent.step(state[agent.firm_id].reshape(1, -1),
                                actions[agent.firm_id],
                                reward,
                                next_state[agent.firm_id].reshape(1, -1),
                                done)
            state = next_state
            if done:
                break
        
        # 更新epsilon
        eps = max(config.eps_end, config.eps_decay * eps)
        
        # 记录DQNAgent分数
        for agent in agents:
            scores[agent.firm_id].append(episode_scores[agent.firm_id])
        
        # 输出进度
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}/{num_episodes} | Epsilon: {eps:.4f}')
            for agent in agents:
                avg_score = np.mean(scores[agent.firm_id][-100:])
                print(f'  Firm {agent.firm_id} Average Score: {avg_score:.2f}')
        
        # 保存DQNAgent模型
        if i_episode % 500 == 0:
            for agent in dqn_agents:
                agent.save(f'models/dqn_agent_firm_{agent.firm_id}_episode_{i_episode}.pth')
    
    # 保存最终DQNAgent模型
    for agent in dqn_agents:
        agent.save(f'models/dqn_agent_firm_{agent.firm_id}_final.pth')
    
    return scores

def test_agent(env, agents, num_episodes=10):
    """
    测试训练好的DQN智能体
    
    :param env: 环境
    :param agents: 训练好的DQN智能体列表
    :param num_episodes: 测试的episodes数量
    :return: 所有episode的奖励和详细信息
    """
    # 为每个agent创建数据结构
    all_scores = [[] for _ in agents]
    all_inventory_history = [[] for _ in agents]
    all_orders_history = [[] for _ in agents]
    all_demand_history = [[] for _ in agents]
    all_satisfied_demand_history = [[] for _ in agents]
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        episode_scores = [0] * len(agents)
        episode_inventories = [[] for _ in agents]
        episode_orders = [[] for _ in agents]
        episode_demands = [[] for _ in agents]
        episode_satisfied_demands = [[] for _ in agents]
        
        for t in range(env.max_steps):
            # 所有企业采取动作
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                # 检查是否是智能体企业
                agent = next(a for a in agents if a.firm_id == firm_id)
                firm_state = state[firm_id].reshape(1, -1)
                action = agent.act(firm_state)
                actions[firm_id] = action

            # 执行动作
            next_state, rewards, done, rshape_rewards = env.step(actions)
            
            # 记录每个智能体的关键指标
            for i, agent in enumerate(agents):
                episode_inventories[i].append(env.inventory[agent.firm_id][0])
                episode_orders[i].append(actions[agent.firm_id][0])
                episode_demands[i].append(env.demand[agent.firm_id][0])
                episode_satisfied_demands[i].append(env.satisfied_demand[agent.firm_id][0])
                episode_scores[i] += rewards[agent.firm_id][0]
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录每个智能体的分数和历史数据
        for i in range(len(agents)):
            all_scores[i].append(episode_scores[i])
            all_inventory_history[i].append(episode_inventories[i])
            all_orders_history[i].append(episode_orders[i])
            all_demand_history[i].append(episode_demands[i])
            all_satisfied_demand_history[i].append(episode_satisfied_demands[i])
        
        print(f'Test Episode {i_episode}/{num_episodes}')
        for i, agent in enumerate(agents):
            print(f'  Firm {agent.firm_id} Score: {episode_scores[i]:.2f}')
    
    return all_scores, all_inventory_history, all_orders_history, all_demand_history, all_satisfied_demand_history

def plot_training_results(all_scores, window_size=100):
    """
    绘制训练结果
    
    :param all_scores: 每个agent的每个episode的奖励 [firm_num, num_episodes]
    :param window_size: 移动平均窗口大小
    """
    # 计算移动平均
    def moving_average(data, window_size):
        return [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
    
    plt.figure(figsize=(10, 6))
    
    for firm_id, scores in enumerate(all_scores):
        if len(scores) == 0:
            continue
            
        avg_scores = moving_average(scores, window_size)
        plt.plot(np.arange(len(scores)), scores, alpha=0.3, label=f'企业{firm_id}原始奖励')
        plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'企业{firm_id}移动平均')
    
    plt.title('DQN训练过程中的奖励')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.legend()
    plt.savefig('figures/training_rewards.png')
    plt.close()

def plot_test_results(all_scores, all_inventory_history, all_orders_history, all_demand_history, all_satisfied_demand_history):
    """
    绘制测试结果
    
    :param all_scores: 每个agent的每个episode的奖励
    :param all_inventory_history: 每个agent的每个episode的库存历史
    :param all_orders_history: 每个agent的每个episode的订单历史
    :param all_demand_history: 每个agent的每个episode的需求历史
    :param all_satisfied_demand_history: 每个agent的每个episode的满足需求历史
    """
    num_agents = len(all_scores)
    
    # 创建图表 - 每个agent一行，4列
    fig, axs = plt.subplots(num_agents, 4, figsize=(20, 5*num_agents))
    
    for i in range(num_agents):
        # 计算当前agent的平均值
        avg_inventory = np.mean(all_inventory_history[i], axis=0)
        avg_orders = np.mean(all_orders_history[i], axis=0)
        avg_demand = np.mean(all_demand_history[i], axis=0)
        avg_satisfied_demand = np.mean(all_satisfied_demand_history[i], axis=0)
        
        # 库存图表
        axs[i, 0].plot(avg_inventory)
        axs[i, 0].set_title(f'企业{i} - 平均库存')
        axs[i, 0].set_xlabel('时间步')
        axs[i, 0].set_ylabel('库存量')
        
        # 订单图表
        axs[i, 1].plot(avg_orders)
        axs[i, 1].set_title(f'企业{i} - 平均订单量')
        axs[i, 1].set_xlabel('时间步')
        axs[i, 1].set_ylabel('订单量')
        
        # 需求和满足需求图表
        axs[i, 2].plot(avg_demand, label='需求')
        axs[i, 2].plot(avg_satisfied_demand, label='满足的需求')
        axs[i, 2].set_title(f'企业{i} - 平均需求 vs 满足的需求')
        axs[i, 2].set_xlabel('时间步')
        axs[i, 2].set_ylabel('数量')
        axs[i, 2].legend()
        
        # 奖励柱状图
        axs[i, 3].bar(range(len(all_scores[i])), all_scores[i])
        axs[i, 3].set_title(f'企业{i} - 测试episode奖励')
        axs[i, 3].set_xlabel('Episode')
        axs[i, 3].set_ylabel('总奖励')
    
    plt.tight_layout()
    plt.savefig('figures/test_results.png')
    plt.close()

if __name__ == "__main__":
    # 创建保存模型和图表的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    np.random.seed(50)

    # 创建智能体列表

    #=============================#
    # 1. 复现四个企业，奖励重塑实验
    #=============================#
    config = Config()
    env = Env(config)
    agents = []
    for firm_id in range(config.num_firms):
    
        
        agent = DQNAgent(firm_id=firm_id, config=config)
        agent.load(os.path.join("models_trained", "reshape_reward", f"firm_{firm_id}_final.pth")) #加载训练好的模型的参数,不用reshape的（改为noreshape_reward）
        agents.append(agent)
    
    # 训练智能体，则不需要加载训练好的参数
    # scores = train_dqn(env, agents, num_episodes=2000, config=config, use_reshape_reward=False)
    # plot_training_results(scores)

    # 如果将DqnAgent变为MoEAgent需要进行拼接
    for firm_id in range(config.num_firms):
         agents[firm_id] = MOEAgent(agents[firm_id], firm_id, config, target = 20)

    # 测试训练好的智能体
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agents, num_episodes=10)
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)


    #=============================#
    # 2. 复现3个企业，采用MOE策略
    #=============================#
    # np.random.seed(5)
    # config = Config()
    # config.num_firms = 3 # 公司数量
    # config.beta = np.array([[0.3], [0.3], [0.3]]) # 奖励重塑系数
    # config.initial_inventory = [20, 100, 100]  # 初始库存
    # config.p = [10, 9, 8]  # 价格列表
    # env = Env(config)
    # agents = []
    # for firm_id in range(config.num_firms):
    #     agent = BSAgent(firm_id=firm_id, config=config, target=config.initial_inventory[firm_id])
    #     agents.append(agent)
    
    # agents[1] = DQNAgent(firm_id=1, config=config)
    # agents[1].load(os.path.join("models_trained", "bs_firm_1_final.pth"))
    # agents[1] = MOEAgent(agents[1], firm_id=1, config=config, target=20)
    
    # # 训练智能体，则不需要加载训练好的参数
    # # scores = train_dqn(env, agents, num_episodes=2000, config=config, use_reshape_reward=False)
    # # plot_training_results(scores)

    # # 测试训练好的智能体
    # test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agents, num_episodes=10)
    # plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)
