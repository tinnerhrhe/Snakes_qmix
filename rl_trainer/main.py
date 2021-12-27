import argparse
import datetime
from tensorboardX import SummaryWriter
from replay_buffer import ReplayBuffer
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from common import *
from log_path import *
from env.chooseenv import make
#from dqnAgent import *
from rollout import CommRolloutWorker
from agent import Qagent
from algo.qmix import QMIX
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def generate_episode(args, agents, env, episode_num=None, evaluate=False):
    #if args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
        #env.close()
    env.reset()
    #agents = Qagent.Agents(env, model, args)
    o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
    terminated = False
    win_tag = False
    step = 0
    episode_reward = 0
    episode_reward1 = np.zeros(6)
    last_action = np.zeros((3, 4))
    #print("------",agents.policy)
    agents.policy.init_hidden(1)
    epsilon = 0 if evaluate else args.epsilon
    if args.epsilon_anneal_scale == 'episode':
        epsilon = epsilon - args.anneal_epsilon if epsilon > args.min_epsilon else epsilon
    state = env.get_all_observes()[0]
    while not terminated and step < args.episode_length:
        # time.sleep(0.2)

        obs = get_observations(state, [0, 1, 2], 46, env.board_height, env.board_width)
        #obs = visual_ob(state[0])
        actions, avail_actions, actions_onehot = [], [], []
        for agent_id in range(3):
            avail_action = get_available_action(state, agent_id)
            #print(avail_action)
            action = agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                avail_action, epsilon, evaluate)
            # generate onehot vector of th action
            action_onehot = np.zeros(4)
            action_onehot[action] = 1
            actions.append(np.int(action))
            actions_onehot.append(action_onehot)
            avail_actions.append(avail_action)
            last_action[agent_id] = action_onehot
        if evaluate:
            print("=====",actions)
        all_actions = transform_actions(state, actions, env.board_height, env.board_width)
        next_state, reward, terminated, _, info = env.step(env.encode(all_actions))
        episode_reward += np.sum(reward[0:3]) - np.sum(reward[3:6])
        episode_reward1 += np.array(reward)
        if terminated:
            if np.sum(episode_reward1[:3]) > np.sum(episode_reward1[3:]):
                step_reward = get_reward(info, [0, 1, 2], reward, score=1)
            elif np.sum(episode_reward1[:3]) < np.sum(episode_reward1[3:]):
                step_reward = get_reward(info, [0, 1, 2], reward, score=2)
            else:
                step_reward = get_reward(info, [0, 1, 2], reward, score=0)
        else:
            if np.sum(episode_reward1[:3]) > np.sum(episode_reward1[3:]):
                step_reward = get_reward(info, [0, 1, 2], reward, score=3)
            elif np.sum(episode_reward1[:3]) < np.sum(episode_reward1[3:]):
                step_reward = get_reward(info, [0, 1, 2], reward, score=4)
            else:
                step_reward = get_reward(info, [0, 1, 2], reward, score=0)
        reward = np.sum(step_reward)

        #reward = np.sum(reward[0:3]) - np.sum(reward[3:6])
        win_tag = True if terminated and info['score'][0:3] > info['score'][3:6] else False#battle_won' in info and info['battle_won'] else False
        o.append(obs)
        #print("----------+",state_to_list(state).shape)
        s.append(state_to_list(state))
        u.append(np.reshape(actions, [3, 1]))
        u_onehot.append(actions_onehot)
        avail_u.append(avail_actions)
        r.append([reward])
        terminate.append([terminated])
        padded.append([0.])

        step += 1
        if args.epsilon_anneal_scale == 'step':
            epsilon = epsilon - args.anneal_epsilon if epsilon > args.min_epsilon else epsilon
        # last obs
        state = next_state[0]
    state = next_state[0]
    obs = get_observations(state, [0, 1, 2], 46, env.board_height, env.board_width)
    o.append(obs)
    #print("-----++",state_to_list(state).shape)
    s.append(state_to_list(state))
    o_next = o[1:]
    s_next = s[1:]
    o = o[:-1]
    s = s[:-1]
    #print("<<<<",len(s),s[0].shape)
    # get avail_action for last obs，because target_q needs avail_action in training
    avail_actions = []
    for agent_id in range(3):
        avail_action = [1,1,1,1]        #self.env.get_avail_agent_actions(agent_id)
        avail_actions.append(avail_action)
    avail_u.append(avail_actions)
    avail_u_next = avail_u[1:]
    avail_u = avail_u[:-1]

    # if step < self.episode_limit，padding
    for i in range(step, args.episode_length):
        o.append(np.zeros((3, 46)))
        u.append(np.zeros([3, 1]))
        #print("======++",np.zeros(args.state_shape).shape)
        s.append(np.zeros(args.state_shape))
        r.append([0.])
        o_next.append(np.zeros((3, 46)))
        s_next.append(np.zeros(args.state_shape))
        u_onehot.append(np.zeros((3, 4)))
        avail_u.append(np.zeros((3, 4)))
        avail_u_next.append(np.zeros((3, 4)))
        padded.append([1.])
        terminate.append([1.])

    episode = dict(o=o.copy(),
                   s=s.copy(),
                   u=u.copy(),
                   r=r.copy(),
                   avail_u=avail_u.copy(),
                   o_next=o_next.copy(),
                   s_next=s_next.copy(),
                   avail_u_next=avail_u_next.copy(),
                   u_onehot=u_onehot.copy(),
                   padded=padded.copy(),
                   terminated=terminate.copy()
                   )
    # add episode dim
    for key in episode.keys():
        episode[key] = np.array([episode[key]])
    if not evaluate:
        args.epsilon = epsilon
    #if evaluate and episode_num == args.evaluate_epoch - 1 and args.replay_dir != '':
    #   env.save_replay()
    #print("==========", episode['s'].shape)
    return episode, episode_reward, win_tag, step
def evaluate(env, agent, args):
    win_number = 0
    episode_rewards = 0
    for epoch in range(args.evaluate_epoch):
        _, episode_reward, win_tag, _ = generate_episode(args, agent, env, epoch, evaluate=True)
        episode_rewards += episode_reward
        if win_tag:
            win_number += 1
    print("winRate=========",win_number / args.evaluate_epoch)
    return win_number / args.evaluate_epoch, episode_rewards / args.evaluate_epoch

def main(args):
    #print("==algo: ", args.algo)
    print(f'device: {device}')
    #print(f'model episode: {args.model_episode}')
    #print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)   #gym snakes3v3
    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 46
    print(f'observation dimension: {obs_dim}')
    win_rates = []
    episode_rewards = []
    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    #model = QMIX(args)
    '''
    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)
    '''
    episode = 0
    train_steps = 0
    agent = Qagent.Agents(env, args)

    buffer = ReplayBuffer(args, args.buffer_size, args.batch_size)
    #agent = DQNAgent(env, args, buffer)
    #while time_steps // 200 < args.episode_length:
    for epoch in range(20000):                      #args.max_episodes
        """
        print('Run {}, time_steps {}'.format(episode, time_steps))
        if time_steps // args.evaluate_cycle > evaluate_steps:
            win_rate, episode_reward = evaluate(env, agent, args)
            # print('win_rate is ', win_rate)
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            #self.plt(num)
            evaluate_steps += 1
        """
        episodes = []
        # 收集self.args.n_episodes个episodes
        for episode_idx in range(10):
            episode, _, _, steps = generate_episode(args, agent, env, episode_idx)
            episodes.append(episode)
            #time_steps += steps
            # print(_)
        # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        #print(episode_batch['s'].shape)
        buffer.store_episode(episode_batch)
        for train_step in range(4):
            mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
            agent.train(mini_batch, train_steps)
            train_steps += 1
        if epoch % 2 == 0:
            win_rate, episode_reward = evaluate(env, agent, args)
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            print("train epoch: {}, win rate: {}%, episode reward: {}".format(epoch, win_rate, episode_reward))
    win_rate, episode_reward = evaluate(env, agent, args)
    print('win_rate is ', win_rate)
    win_rates.append(win_rate)
    print(win_rates)
    print(episode_rewards)
    episode_rewards.append(episode_reward)
    #self.plt(num)
'''
    while episode < args.max_episodes:


        # Receive initial observation state s1
        state = env.reset()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:

            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)

            # ============================== add opponent actions =================================
            # we use rule-based greedy agent here. Or, you can switch to random agent.
            actions = logits_greedy(state_to_training, logits, height, width)   #?
            # actions = logits_random(act_dim, logits)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions)) #?step
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)

            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=4)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)

            done = np.array([done] * ctrl_agent_num)

            # ================================== collect data ========================================
            # Store transition in R
            model.replay_buffer.push(obs, logits, step_reward, next_obs, done)

            model.update()

            obs = next_obs
            step += 1

            if args.episode_length <= step or (True in done):

                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} epsilon: {model.eps:.2f}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')

                reward_tag = 'reward'
                loss_tag = 'loss'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                    'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})
                if model.c_loss and model.a_loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'actor': model.a_loss, 'critic': model.c_loss})

                if model.c_loss and model.a_loss:
                    print(f'\t\t\t\ta_loss {model.a_loss:.3f} c_loss {model.c_loss:.3f}')

                if episode % args.save_interval == 0:
                    model.save_model(run_dir, episode)

                env.reset()
                break
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="qmix", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    #parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.01, type=float)
    parser.add_argument('--c_lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=1000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    #parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)
    parser.add_argument('--model_dir', type=str, default='./agent', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--replay_dir', type=str, default='./replay', help='absolute path to save the replay')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = True
    args.hyper_hidden_dim = 128
    args.qtran_hidden_dim = 64
    args.lr = 5e-3
    args.episode_limit = args.episode_length
    # epsilon greedy
    args.epsilon = 0.95
    args.min_epsilon = 0.05
    anneal_steps = 500
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'episode'
    args.n_agents = 3
    args.n_actions = 4
    args.state_shape = 206
    # the number of the train steps in one epoch
    args.train_steps = 200

    # experience replay
    args.batch_size = 4
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 20
    # how often to update the target_net
    args.target_update_cycle = 100


    # prevent gradient explosion
    args.grad_norm_clip = 10

    main(args)
