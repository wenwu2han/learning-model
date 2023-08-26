import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# 定义简化的自动化仓库环境（多智能体版本）
class WarehouseEnvironment:
    def __init__(self, num_agents, num_states, num_actions):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.states = [0] * num_agents
        self.target_state = [num_states - 1] * num_agents
        self.done = [False] * num_agents

    def reset(self):
        self.states = [0] * self.num_agents
        self.done = [False] * self.num_agents
        return self.states

    def step(self, actions):
        rewards = [0] * self.num_agents

        for i in range(self.num_agents):
            if self.states[i] == self.target_state[i]:
                self.done[i] = True

            if actions[i] == 1 and self.states[i] < self.num_states - 1:
                self.states[i] += 1

            rewards[i] = 1 if self.states[i] == self.target_state[i] else 0

        return self.states, rewards, self.done


# 定义 Q-网络模型
class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 定义多智能体 Q-学习算法
class MultiAgentQLearning:
    def __init__(self, num_agents, num_states, num_actions, learning_rate, gamma, epsilon):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_networks = [QNetwork(num_states, num_actions) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(network.parameters(), lr=learning_rate) for network in self.q_networks]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return [random.randint(0, self.num_actions - 1) for _ in range(self.num_agents)]
        else:
            q_values = [network(torch.FloatTensor([state[i]])) for i, network in enumerate(self.q_networks)]
            return [q.argmax().item() for q in q_values]

    def update_q_values(self, states, actions, rewards, next_states):
        for i in range(self.num_agents):
            q_values = self.q_networks[i](torch.FloatTensor([states[i]]))
            next_max_q = torch.max(self.q_networks[i](torch.FloatTensor([next_states[i]])))
            target_q = rewards[i] + self.gamma * next_max_q
            loss = nn.MSELoss()(q_values[0, actions[i]], target_q)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()


# 主函数
def main():
    num_agents = 3
    num_states = 5
    num_actions = 2
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000

    env = WarehouseEnvironment(num_agents, num_states, num_actions)
    q_learning_agents = [MultiAgentQLearning(num_agents, num_states, num_actions, learning_rate, gamma, epsilon) for _
                         in range(num_agents)]

    for episode in range(num_episodes):
        states = env.reset()
        done = False

        while not all(done):
            actions = [q_learning_agents[i].select_action(states) for i in range(num_agents)]
            next_states, rewards, done = env.step(actions)

            for i in range(num_agents):
                q_learning_agents[i].update_q_values(states[i], actions[i], rewards[i], next_states[i])

            states = next_states

        if episode % 100 == 0:
            print(f"Episode {episode}")


if __name__ == "__main__":
    main()
