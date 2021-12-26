import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map
def make_grid_map1(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        head = True
        for p in pos:
            if head:
                if 2<= index <=4:
                    snakes_map[p[0]][p[1]][0] = index + 10
                else:
                    snakes_map[p[0]][p[1]][0] = index + 20
                head = False
            else:
                if 2<= index <=4:
                    snakes_map[p[0]][p[1]][0] = index + 30
                else:
                    snakes_map[p[0]][p[1]][0] = index + 40

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
#16+10*3=46
def get_observations(state, agents_index, obs_dim, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0:3] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations

def visual_ob(state):
    image = np.zeros((20, 10))
    for i in range(7):
        snake_i = state[i+1] #[[7, 0], [0, 0], [7, 17], [0, 16], [3, 5]]
        for cordinate in snake_i:#[7, 0]
            image[cordinate[1]][cordinate[0]] = i+1
    return image
def visual_obs(state):
    image = np.zeros((20, 10))
    for i in range(7):
        snake_i = state[i+1] #[[7, 0], [0, 0], [7, 17], [0, 16], [3, 5]]
        for cordinate in snake_i:#[7, 0]
            image[cordinate[1]][cordinate[0]] = i+1
    return image
def logits_AC(state, logits, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    #logits = torch.Tensor(logits).to(device)
    logits = np.trunc(logits)
    logits_action = np.array([out for out in logits])

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list
def get_reward_ppo(info, snake_index, reward, score):
    # print(info['snakes_position'])
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))

    ###关于长度
    for i in snake_index:
        # 周围距离
        self_head = np.array(snake_heads[i])
        dists_bean = [np.sqrt(np.sum(np.square(beans_head - self_head))) for beans_head in beans_position]
        dists_body = []
        for j in range(6):
            if j != i:
                dists_body = [np.sqrt(np.sum(np.square(np.array(snakes_body) - np.array(snake_heads[i]))))
                              for snakes_body in snakes_position]
        if score == 1:  # 结束AI赢
            step_reward[i] += 0.05
        elif score == 2:  # 结束random赢
            step_reward[i] -= 0.05
        elif score == 0:  # 平 一样长
            step_reward[i] = 0

        # if min(dists_body) >= 1:
        elif score == 3:  # 未结束AI长
            step_reward[i] += 0.02
        elif score == 4:  # 未结束random长
            step_reward[i] -= 0.02

        ###关于吃豆
        if reward[i] > 0:  # 吃到
            step_reward[i] += 0.04
        else:  # 没吃到看距离
            # if min(dists_body) >= 1:
            step_reward[i] -= max(min(dists_bean) / 1000 - 0.002, 0)  # 0.027 min(dists_bean)/1000
            if reward[i] < 0:
                step_reward[i] -= 0.02

    return step_reward * 10


def get_reward(info, snake_index, reward, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 25
        elif score == 3:
            step_reward[i] += 10
        elif score == 4:
            step_reward[i] -= 5

        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            step_reward[i] -= min(dists)
            if reward[i] < 0:
                step_reward[i] -= 10

    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions
def get_available_action(state, id):
    state_copy = state.copy()
    beans = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list
    width = state_copy['board_width']
    height = state_copy['board_height']
    beans_position = copy.deepcopy(beans)
    actions = []
    head_x = snakes[id][0][1]
    head_y = snakes[id][0][0]
    snake_map = make_grid_map(width, height, beans, snakes_positions)
    state_ = np.array(snake_map)
    state_ = np.squeeze(state_, axis=2)
    head_surrounding = get_surrounding(state_, width, height, head_x, head_y)
    bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
    beans_position.pop(index)

    next_distances = []
    up_distance = math.inf if head_surrounding[0] > 1 else \
        math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
    next_distances.append(up_distance)
    down_distance = math.inf if head_surrounding[1] > 1 else \
        math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
    next_distances.append(down_distance)
    left_distance = math.inf if head_surrounding[2] > 1 else \
        math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
    next_distances.append(left_distance)
    right_distance = math.inf if head_surrounding[3] > 1 else \
        math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
    next_distances.append(right_distance)
    index = (next_distances.index(max(next_distances)))
    for i in range(4):
        if i == index:
            actions.append(0)
        else:
            actions.append(1)
    return actions
def t_state(state):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    feature = np.zeros((7, 10, 20))
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map1(board_width, board_height, beans_positions, snakes_positions)
    for i in range(board_width):
        for j in range(board_height):
            feature[0][i][j] = 1 if snake_map[i][j][0] >= 30 else 0
            feature[1][i][j] = 1 if 30 < snake_map[i][j][0] < 40 else 0
            feature[2][i][j] = 1 if snake_map[i][j][0] > 40 else 0
            feature[3][i][j] = 1 if 10 < snake_map[i][j][0] < 30 else 0
            feature[4][i][j] = 1 if 10 < snake_map[i][j][0] < 20 else 0
            feature[5][i][j] = 1 if 20 < snake_map[i][j][0] < 30 else 0
            feature[6][i][j] = 1 if snake_map[i][j][0] == 1 else 0
    return feature
def logits_greedy(state, logits, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list
def transform_actions(state, actions, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list
    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = actions
    action_list[3:] = greedy_action

    return action_list
def state_to_list(states):
    state_copy = states.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = []
    name = {"up": 0, "down": 1, "left": 2, "right": 3}
    '''
    for key, value in states.items():
        if isinstance(key, int):
            num = 0
            for _ in value:
                state.append(_[0] / 10)
                state.append(_[1] / 20)
                num += 1
            for k in range(num):
                state.append(1)
        else:
            if key == 'last_direction':
                while len(state) < 600:
                    state.append(0.)
                if value is None:
                    for i in range(6):
                        state.append(0)
                else:
                    for _ in value:
                        state.append(name[_])
                        #state.append(np.eye(4)[name[_]])
    state = np.array(state)
    state = state.astype(dtype=np.float32)
    '''
    snake_map = list(np.array(snake_map).flatten())
    state = snake_map
    if state_copy['last_direction'] is None:
        for i in range(6):
            state.append(0)
    else:
        for k in state_copy['last_direction']:
            state.append(name[k])
    state = np.array(state)
    state = state.astype(dtype=np.float32)
    #state = state.flatten()
    #state = state.astype(dtype=np.float32)
    #print("++++++++++++++",state.shape)
    #print("+++++",state.shape)
    return state
def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()