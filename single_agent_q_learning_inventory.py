import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# 定义简化的自动化仓库环境
class WarehouseEnvironment:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state = 0  # 当前状态
        self.target_state = num_states - 1  # 目标状态
        self.done = False

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.state == self.target_state:
            self.done = True

        if action == 1 and self.state < self.num_states - 1:
            self.state += 1

        reward = 1 if self.state == self.target_state else 0
        return self.state, reward, self.done


# 定义 Q-网络模型
class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 10)
        self.fc2 = nn.Linear(10, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练 Q-学习算法
def q_learning(env, q_network, num_episodes, learning_rate, gamma, epsilon):
    optimizer = optim.SGD(q_network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            q_values = q_network(torch.tensor(state, dtype=torch.float32))
            action = select_action(q_values, epsilon)
            next_state, reward, done = env.step(action)

            next_q_values = q_network(torch.tensor(next_state, dtype=torch.float32))
            target_q = reward + gamma * torch.max(next_q_values)
            predicted_q = q_values[action]

            loss = criterion(predicted_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item()}")


# 选择动作（ε-贪心策略）
def select_action(q_values, epsilon):
    if random.random() < epsilon:
        return random.randint(0, q_values.size(0) - 1)
    else:
        return torch.argmax(q_values).item()


# 主函数
def main():
    num_states = 5  # 状态数（示例中使用 5 个状态）
    num_actions = 2  # 动作数（示例中使用 2 个动作）
    learning_rate = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000

    env = WarehouseEnvironment(num_states, num_actions)
    q_network = QNetwork(num_states, num_actions)

    q_learning(env, q_network, num_episodes, learning_rate, gamma, epsilon)


if __name__ == "__main__":
    main()
