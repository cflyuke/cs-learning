import numpy as np

class Env:
    def __init__(self, config):
        self.num_firms = config.num_firms
        self.p = config.p  # 企业的价格列表
        self.h = config.h  # 库存持有成本
        self.c = config.c  # 损失销售成本
        self.poisson_lambda = config.poisson_lambda  # 泊松分布的均值
        self.max_steps = config.max_steps  # 每个episode的最大步数
        self.initial_inventory = config.initial_inventory
        
        # 初始化库存
        self.inventory = np.array(config.initial_inventory).reshape(config.num_firms, 1)
        # 初始化订单量
        self.orders = np.zeros((config.num_firms, 1))
        # 初始化已满足的需求量
        self.satisfied_demand = np.zeros((config.num_firms, 1))
        # 初始化从供应商那里得到的
        self.arrived_order = np.zeros((config.num_firms, 1))
        # 初始化收到的订单
        self.demand = np.zeros((config.num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记episode是否结束
        self.done = False
        # 添加奖励重塑系数
        self.beta = config.beta

    def reset(self):
        """
        重置环境状态。
        """
        self.inventory = np.array(self.initial_inventory).reshape(self.num_firms, 1) 
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.arrived_order = np.zeros((self.num_firms, 1)) 
        self.demand = np.zeros((self.num_firms, 1))

        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的观察信息，包括订单量、满足的需求量和库存。
        每个企业的状态是独立的，包括自己观察的订单、需求和库存。
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory, self.arrived_order, self.demand), axis=1)

    def _generate_demand(self):
        """
        根据规则生成每个企业的需求。
        最下游企业的需求遵循泊松分布，其他企业的需求等于下游企业的订单量。
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循泊松分布，均值为 poisson_lambda
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]  # d_{i+1,t} = q_{it}
        return demand

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动 (每个企业的订单量) 更新环境状态。
        
        :param actions: 每个企业的订单量 (shape: (num_firms, 1))，即每个智能体的行动
        :return: next_state, reward, done
        """
        self.orders = actions  # 更新订单量
        
        # 生成各企业的需求
        self.demand = self._generate_demand()

        # 计算每个企业收到的订单量和满足的需求
        for i in range(self.num_firms):
            self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
            self.arrived_order[i] = min(self.orders[i], self.inventory[i+1]) if i < self.num_firms - 1 else self.orders[i]
        
        # 更新库存
        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]
        
        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))  # 损失销售费用
        
        for i in range(self.num_firms):
            rewards[i] += self.p[i] * self.satisfied_demand[i] - (self.p[i+1] if i+1 < self.num_firms else 0) * self.orders[i] - self.h * self.inventory[i]
            
            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c
        
        rewards -= loss_sales  # 总奖励扣除损失销售成本
        
        reshape_rewards = rewards - self.beta / 3 * (rewards - rewards.mean())
        # 增加步数
        self.current_step += 1
        
        # 判断是否结束（比如达到最大步数）
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self._get_observation(), rewards, self.done, reshape_rewards

