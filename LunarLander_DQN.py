import gym
import numpy as np
from math import sqrt
import torch as t
from torch import nn
from torch import optim
import random
from matplotlib import pyplot as plt
import math
import Box2D

#Define agent class:
class NeilAmstrong:
  def __init__(self, max_epsilon, min_epsilon,epsilon_decay_step,batch_size, C,
               gamma, maximum_memory, replay_start_size, cyclical_learning, state_space, action_space):
    self.max_epsilon = max_epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay_step = epsilon_decay_step
    self.batch_size = batch_size
    self.C = C
    self.gamma = gamma
    self.maximum_memory = maximum_memory
    self.replay_start_size = replay_start_size
    self.cyclical_learning = cyclical_learning
    self.state_space = state_space
    self.action_space = action_space

    self.policy_network = nn.Sequential(nn.Linear(self.state_space, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 84),
                                        nn.ReLU(),
                                        nn.Linear(84, self.action_space))
    self.target_network = nn.Sequential(nn.Linear(self.state_space, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 84),
                                        nn.ReLU(),
                                        nn.Linear(84, self.action_space))
    self.loss_function = nn.MSELoss()
    self.memory = []
    if (not self.cyclical_learning): self.optimize = optim.SGD(self.policy_network.parameters(), lr=0.001)

  def choose_action(self,state,step): #step and state is input from outside
    epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1. * (step-self.replay_start_size) / self.epsilon_decay_step)
    x = np.random.uniform(0,1)
    if (x < epsilon):
      action = np.random.randint(self.action_space)
    else:
      input = t.from_numpy(state).float()
      output = self.policy_network.forward(input)
      action = output.detach().numpy().argmax()
    return action

  def act_greedy(self, state):
    input = t.from_numpy(state).float()
    output = self.policy_network.forward(input)
    action = output.detach().numpy().argmax()
    return action

  def store_memory(self,experience):
    if (len(self.memory) == self.maximum_memory): self.memory.remove(self.memory[0])
    self.memory.append(experience)

  def recall_memory(self):
    if (len(self.memory) > self.batch_size):
      dream = np.array(random.sample(self.memory,self.batch_size)).reshape(self.batch_size,5)
    else: dream = np.array(self.memory).reshape(len(self.memory),5)
    return dream

  def update_cyclical_learning(self, alpha, momentum):
    self.optimize = optim.SGD(self.policy_network.parameters(), lr = alpha, momentum = momentum)

  def learn(self,mini_batch,step):
    s_batch = np.vstack(mini_batch[:, 0]).astype(np.float32)
    s_batch = t.from_numpy(s_batch)
    a_batch = np.vstack(mini_batch[:, 1]).astype(np.int64)
    a_batch = t.from_numpy(a_batch)
    next_s_batch = np.vstack(mini_batch[:, 3]).astype(np.float32)
    next_s_batch = t.from_numpy(next_s_batch)
    r_batch = np.vstack(mini_batch[:, 2]).astype(np.float32)
    r_batch = t.from_numpy(r_batch)
    done_batch = np.vstack(mini_batch[:, 4]).astype(np.bool_)

    self.optimize.zero_grad()

    with t.no_grad(): target_val = self.target_network.forward(next_s_batch)
    max_target_net_val, best_action = t.max(target_val,dim = 1)
    target_net_val = r_batch + self.gamma*max_target_net_val.view(len(max_target_net_val),1)
    for i in range(len(done_batch)):
      if (done_batch[i]): target_net_val[i] = r_batch[i]
    policy_net_val = self.policy_network.forward(s_batch).gather(1,a_batch)
    loss = self.loss_function(target_net_val,policy_net_val)
    loss.backward()
    self.optimize.step()
    if (step % self.C == 0):
      self.target_network.load_state_dict(self.policy_network.state_dict())

#Setting up environment:
env = gym.make('LunarLander-v2')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

ship = NeilAmstrong(max_epsilon=1,min_epsilon=0.001, epsilon_decay_step=50000,batch_size=32, C=2000, gamma=0.99,maximum_memory=10000,replay_start_size=1000, cyclical_learning=True,state_space=state_space,action_space=action_space)
#ship.C = 6000
print(ship.C)
n_episodes = 1000
episode_reward = []
episode_list = []
score_board = []
ep = []
cyclical_step = 2000 #Stepsize for cyclical learning
sum_reward = 0
count_average = 0
step = 0
delta = 1
alpha = 0.001
momentum = 0.95
for i in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while (not done):
        step = step + 1

        #Cyclical Learning Rate:
        if (step % cyclical_step == 0): delta = (-1)*delta
        alpha = alpha + delta*(0.01 - 0.001)/cyclical_step
        momentum = momentum - delta*(0.95 - 0.8)/cyclical_step
        ship.update_cyclical_learning(alpha,momentum)

        if (step < ship.replay_start_size):
            action = np.random.randint(action_space)
            next_state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            reward = reward/10
            experience = [state, action, reward, next_state, done]
            ship.store_memory(experience)
        else:
            action = ship.choose_action(state,step)
            next_state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            reward = reward/10
            exprience = [state, action, reward, next_state, done]
            ship.store_memory(exprience)
            mini_batch = ship.recall_memory()

            ship.learn(mini_batch,step)
        state = next_state
    print("This is episode:", i, ". | Total reward is:",total_reward)
    episode_reward.append(total_reward)
    episode_list.append(i)

    #Record Performance
    sum_reward = sum_reward + total_reward
    if (count_average == 9):
        average_reward = sum_reward/10
        score_board.append(average_reward)
        ep.append(i)
        sum_reward = 0
        count_average = 0
    else: count_average = count_average + 1

#Create graph for training performance
plt.plot(episode_list,episode_reward, label = 'Episode reward')
plt.plot(ep,score_board, label = 'Mean reward for last 10 episodes')

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.show()

#Test Trained Agent:
print("--Testing--")
#Test
ep_test = []
total_reward_record = []
for i in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    while (not done):
        action = ship.act_greedy(state)
        next_state, reward, done, info = env.step(action)
        total_reward = total_reward + reward #accumulating total reward
        state = next_state
    total_reward_record.append(total_reward)
    ep_test.append(i+1)
    print(i+1,total_reward)

plt.plot(ep_test,total_reward_record, label = 'Episode reward')
plt.plot(ep_test, [sum(total_reward_record)/len(total_reward_record)]*len(total_reward_record), label = 'Episode mean')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.show()

