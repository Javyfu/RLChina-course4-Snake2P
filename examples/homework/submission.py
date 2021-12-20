# # This is homework.
# # Load your model and submit this to Jidi
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optimizer

# load critic
from pathlib import Path
import sys
class Critic_CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer


        self.conv1 = nn.Sequential(  # input shape (1,6,8)
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=8,  # n_filter
                      kernel_size=2,  # filter size
                      stride=1,  # filter step
                      padding='same'  # con2d出来的图片大小不变
                      ),  # output shape (8,28,28)
            nn.MaxPool2d(kernel_size=2)  # 2x2采样，output shape (8,3,4)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 2, 1, 'same'),  # output shape (16,3,4)
                                   nn.ReLU())

        self.linear_in = nn.Linear(16 * 3 * 4, hidden_size)

        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)

        self.linear_out = nn.Linear(hidden_size, self.output_size)


    def forward(self, x):
        #print(x,x.shape)
        x = self.conv1(x)
        #print(x,x.shape)
        x = self.conv2(x)
        #print(x, x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear_in(x)
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        return x

# TODO
class IQL:
    def __init__(self):

        self.state_dim = 48
        self.action_dim = 4

        self.hidden_size = 64

        self.critic_eval = Critic_CNN(self.state_dim, self.action_dim, self.hidden_size)


    def choose_action(self, observation):
        inference_output = self.inference(observation)
        return inference_output

    def inference(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, 1, 6, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return {"action": action}

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))


#TODO
def action_from_algo_to_env(joint_action):
    pass


# todo
# Once start to train, u can get saved model. Here we just say it is critic.pth.
critic_net = os.path.dirname(os.path.abspath(__file__)) + '/critic_0_2000.pth'
agent = IQL()
agent.load(critic_net)


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def get_observations(state, id):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    state[state == 2+id] = 6
    state[state == 2] = 4
    state[state == 3] = 4


    observations = state.reshape((1,6,8))

    return observations.squeeze().tolist()

def get_joint_action_eval(all_observes):
    joint_action = []
    for i in range(2):
        agent_id = i
        a_obs = all_observes[agent_id]
        each = choose_action_to_env(a_obs, agent_id)
        joint_action.append(each)
    return joint_action

def choose_action_to_env(observation, id):
    obs_copy = observation.copy()
    action_from_algo = agent.choose_action(obs_copy)
    action_to_env = action_from_algo_to_env(action_from_algo)
    return action_to_env

def action_from_algo_to_env(joint_action):
        '''
        :param joint_action:
        :return: wrapped joint action: one-hot
        '''
        joint_action_ = [0,0,0,0]
        action_a = joint_action["action"]
        joint_action_[action_a] = 1
        return joint_action_

# todo
def my_controller(observation, action_space, is_act_continuous=False):
    obs_all = [get_observations(observation, id) for id in range(2)]

    action_all = get_joint_action_eval(obs_all)
    return action_all