import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
parser.add_argument('--gamma', type=float, default=0, metavar='G',
                    help='discount factor (default: 0)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make("CartPole-v1", render_mode="human")
env.reset(seed=args.seed) #set seed of the env
torch.manual_seed(args.seed) #set seed of the torch


class Policy(nn.Module): 
    def __init__(self):
        """Policy network

        DNN model consisting of 2 linear layers with ReLU activation
        
        """
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128) #linear layer taking (4,) input and outputting (128,) output
        self.dropout = nn.Dropout(p=0.6) #dropout 60% of the neurons
        self.affine2 = nn.Linear(128, 2) #linear layer taking (128,) input and outputting (2,) output such that 0 = L, 1 = R

        self.saved_log_probs = [] #list to save log probabilities of actions
        self.rewards = [] #list to save rewards

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1) #return softmax of action scores with output shape (2,)


policy = Policy() #initialize policy model
optimizer = optim.Adam(policy.parameters(), lr=1e-2) #initialize optimizer
eps = np.finfo(np.float32).eps.item() #small value to avoid division by zero


def select_action(state):
    """Select action based on policy

    Args:
        state (torch): state of the environment

    Returns:
        int: action to take
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    """Finish episode"""
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()

#to show window with live render run (will run slower than without render): 
#python reinforce.py  --render

#comment to initiate gene's branch
