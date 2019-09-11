#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import factorial


# # For grid World

# In[97]:


def calculate(state, action):
    #From A to A'
    if state == (0, 1): 
        next_state = (4,1)
        r = 10
        return next_state,r
    #From B to B'
    if state == (0, 3): 
        next_state = (2,3)
        r = 5
        return next_state,r
    #Normal movements with 0 reward
    next_state = (state[0] + action[0], state[1] + action[1])
    if ((next_state[0] >= 0 and next_state[0] < 5) and
        (next_state[1] >= 0 and next_state[1] < 5)):
        r=0
        return next_state, r
    #If Wall
    r = -1
    return state, -1 


# # Q2

# In[5]:


actions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
epochs = 90
gamma = 0.9    #given in grid world example
#since there are 4 actions possible, and each action is equi probable
prob = 0.25
v_pi = np.zeros((5, 5))
while epochs>0:
    v_n = np.zeros((5, 5))
    #i and j represents all the states
    for i in range(5):
        for j in range(5):
            #For each action
            for a in actions:
                state, reward = calculate((i, j), a)
                v_n[i, j] = v_n[i, j] + prob * (reward + gamma * v_pi[state[0], state[1]])
    v_pi = v_n
    if epochs % 20 == 0: 
        print(v_pi, '\n-------')
    epochs = epochs - 1


# # Q4

# In[8]:


epochs = 90
v_pi = np.zeros((5, 5))
while epochs>0:
    #For each state
    for i in range(5):
        for j in range(5):
            v = []
            #For each action
            for a in actions:
                state, reward = calculate((i, j), a)
                v.append(reward + gamma * v_pi[state[0], state[1]])
            #Select the optimal value
            v_pi[i, j] = np.max(v)
    if epochs % 20 == 0: 
        print(v_pi, '\n-------')
        print(v, '\n-------')
    epochs = epochs - 1


# # EX : 4.1

# In[9]:


class grid_world():
    def __init__(self,gamma=1,theta=0.01,policy=0.25):
        self.gamma = gamma
        self.theta = theta
        self.policy = policy
        self.actions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        self.value = np.zeros((4, 4))
        
    def evaluate(self, state, action):
        next_state = [state[0] + action[0], state[1] + action[1]]
        if ((next_state[0] >= 0 and next_state[0] < 4) and (next_state[1] >= 0 and next_state[1] < 4)):
            return next_state, -1
        #Wall : cant take any action
        return state, -1 
    
    def start_execution(self):
        itr = 1
        while True:
            #Policy Evaluation
            while True:
                delta = 0 
                for i in range(4):
                    for j in range(4):
                        if (i,j) == (0,0) or (i,j) == (3,3):
                            continue
                        v_prev = self.value[i, j]
                        tmp = 0
                        for a in self.actions:
                            next_state, reward = self.evaluate([i,j], a)
                            tmp = tmp + (self.policy * (reward + self.gamma * self.value[next_state[0], next_state[1]]))
                        self.value[i, j] = tmp
                        delta = max(delta, abs(self.value[i, j]-v_prev))
                if delta < self.theta:
                    break
            #Policy Improvement
            policy_stable = True
            policy_changes = 0
            for i in range(0,4):
                for j in range(0,4):
                    action_returns = []
                    old_action = self.policy
                    for a in self.actions:
                        if (0 <= a[0] <= i) or (j >= abs(a[1]) > 0):
                            s,r = self.evaluate([i, j], a)
                            action_returns.append(r)
                        else:
                            action_returns.append(0)
                    self.policy = self.actions[np.argmax(action_returns)]
                    if old_action != self.policy:
                        policy_stable = False
                        policy_changes = policy_changes+1
            #print('iteration ',itr)
            #print('policies changed= ',policy_changes)
            if policy_stable:
                break
            itr = itr + 1
        #print('Updated Value')
        print(self.value)
        #print('Updated Policy')
        #print(self.policy)
        
g = grid_world()
g.start_execution()


# # EX : 4.7

# In[58]:


class car_rental():
    def __init__(self, gamma=0.9, theta=1e-4):
        self.gamma = gamma
        self.theta = theta
        self.action = np.arange(-5, 6)
        self.policy = np.zeros((21, 21), dtype=int)
        self.value = np.zeros((21, 21))
        self.prob = []
        for i in range(0,8):
            for j in range(0,10):
                for k in range(0,8):
                    for l in range(0,6):
                        self.prob.append(self.poisson(i, 3)*self.poisson(j, 4)*self.poisson(k, 3)*self.poisson(l, 2))
        self.states = [(i, j) for i in range(21) for j in range(21)]
    def poisson(self, n, l):
        return np.exp(-l) * pow(l, n) / factorial(n)

    def evaluate(self,state, r, v):
        # shuttle one car
        if r > 0: 
            returns = 2 
        returns = (-2) * abs(r)
        state = (min(state[0] - r, 20), min(state[1] + r, 20))
        reward = []
        for i in range(0,8):
            for j in range(0,10):
                for k in range(0,8):
                    for l in range(0,6):
                        if (min(state[0] - min(state[0], i) + k, 20) > 10 or min(state[1] - min(state[1], j) + l, 20) > 10):
                            reward.append((min(state[0], i) + min(state[1], j))*10 + 4)
                        else:
                            #Penalty for more than 10 cars
                            reward.append((min(state[0], i) + min(state[1], j))*10)
        reward = np.array(reward)
        new_state = []
        for i in range(0,8):
            for j in range(0,10):
                for k in range(0,8):
                    for l in range(0,6):
                        new_state.append((min(state[0] - min(state[0], i) + k, 20), min(state[1] - min(state[1], j) + l, 20)))
        new_state = np.array(new_state)
        returns = returns + np.sum(self.prob * (reward + self.gamma * v[new_state[:,0], new_state[:,1]]))
        return returns

    def start_execution(self):
        itr = 0
        while 1:
            #Policy evaluation
            while 1:
                delta = 0
                for state in self.states:
                    v_prev = self.value[state[0], state[1]]
                    self.value[state[0], state[1]] = self.evaluate([state[0], state[1]], self.policy[state[0], state[1]], self.value)
                    delta = max(delta, abs(self.value[state[0], state[1]]-v_prev))
                if delta < self.theta:
                    break
            #Policy Improvement
            policy_stable = True
            policy_changes = 0
            for state in self.states:
                action_returns = []
                old_action = self.policy[state[0], state[1]]
                for a in self.action:
                    if (0 <= a <= state[0]) or (state[1] >= abs(a) > 0):
                        action_returns.append(self.evaluate([state[0], state[1]], a, self.value))
                    else:
                        action_returns.append(-np.Infinity)
                self.policy[state[0], state[1]] = self.action[np.argmax(action_returns)]
                if old_action != self.policy[state[0], state[1]]:
                    policy_stable = False
                    policy_changes = policy_changes+1
            print('iteration ',itr)
            print('policies changed= ',policy_changes)
            if policy_stable:
                break
            itr = itr + 1
        print('Updated Value')
        print(self.value)
        print('Updated Policy')
        print(self.policy)


# In[59]:


c = car_rental()
c.start_execution()


# In[ ]:




