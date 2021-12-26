import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.PPO import PPO
from common import *
from log_path import *
from env.chooseenv import make
from plot import cross_loss_curve
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Memory:
    def __init__(self):
        self.m_obs = []
        self.m_obs_next = []

    def clear_memory(self):
        del self.m_obs[:]
        del self.m_obs_next[:]


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)

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
    obs_dim = 26
    print(f'observation dimension: {obs_dim}')

    history_reward = []
    history_step_reward = []
    history_a_loss = []
    history_c_loss = []
    total_step_reward = 0

    memory = Memory()
    Memory_size = 4

    training_stage = 40

    torch.manual_seed(args.seed)

    sample_lr = [
        0.0001, 0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003,
        0.00002, 0.00001, 0.000009, 0.000008, 0.000007, 0.000006, 0.000005,
        0.000004, 0.000003, 0.000002, 0.000001]
    new_lr = 0.0001

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = PPO(obs_dim * Memory_size, act_dim, ctrl_agent_num, args)

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    while episode < args.max_episodes:
        if episode > training_stage:
            try:
                new_lr = sample_lr[int(episode // training_stage)]
            except(IndexError):
                new_lr = 0.000001  # * (0.9 ** ((episode-Decay) //training_stage))

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
        obs = visual_ob(state[0]) / 10

        # Memory-beginning
        for _ in range(Memory_size):
            memory.m_obs.append(obs)
        obs = np.stack(memory.m_obs)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:
            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)
            # print("logits: ",logits)

            # ============================== add opponent actions =================================
            # we use rule-based greedy agent here. Or, you can switch to random agent.
            actions = logits_AC(state_to_training, logits, height, width)
            # print("actions: ",actions)
            # actions = logits_random(act_dim, logits)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions))

            next_state_to_training = next_state[0]
            next_obs = visual_ob(
                next_state_to_training) / 10  # get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)/10

            # Memory
            if len(memory.m_obs_next) != 0:
                del memory.m_obs_next[:1]
                memory.m_obs_next.append(next_obs)
            else:
                memory.m_obs_next = memory.m_obs
                memory.m_obs_next[Memory_size - 1] = next_obs

            next_obs = np.stack(memory.m_obs_next)

            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            if done:  # 结束
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):  # AI赢
                    step_reward = get_reward_ppo(info, ctrl_agent_index, reward, score=1)
                    win_tag = True
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):  # random赢
                    step_reward = get_reward_ppo(info, ctrl_agent_index, reward, score=2)
                    win_tag = False
                else:
                    step_reward = get_reward_ppo(info, ctrl_agent_index, reward, score=0)  # 平
                    win_tag = False
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):  # AI长
                    step_reward = get_reward_ppo(info, ctrl_agent_index, reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):  # random长
                    step_reward = get_reward_ppo(info, ctrl_agent_index, reward, score=4)
                else:  # 一样长
                    step_reward = get_reward_ppo(info, ctrl_agent_index, reward, score=0)

            total_step_reward += sum(step_reward)

            done = np.array([done] * ctrl_agent_num)

            model.memory.rewards.append(step_reward)
            model.memory.is_terminals.append(done)

            if step != 0 and step % 100 == 0:
                model.update(new_lr)
                model.memory.clear_memory()

            # ================================== collect data ========================================
            # Store transition in R
            # model.replay_buffer.push(obs, logits, step_reward,next_obs, done) #[obs,obs,obs][next_obs,next_obs,next_obs]

            obs = next_obs
            step += 1

            if args.episode_length <= step:  # or (True in done)
                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} epsilon: {model.eps:.2f}')
                print(f'=====>>>>>Win or loss:{win_tag}')
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

                history_reward.append(np.sum(episode_reward[0:3]))
                history_a_loss.append(model.a_loss / 100)
                history_c_loss.append(model.c_loss / 10)
                history_step_reward.append(total_step_reward / 1000)  # 10

                model.a_loss = 0
                model.c_loss = 0
                total_step_reward = 0

                cross_loss_curve(history_reward, history_a_loss, history_c_loss, history_step_reward)

                env.reset()
                memory.clear_memory()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ppo", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=1000, type=int)  # 50000
    parser.add_argument('--episode_length', default=2000, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(6e4), type=int)  # 1e5
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)  # 0.0001
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.993, type=float)  # 0.99998

    parser.add_argument("--save_interval", default=20, type=int)  # 1000
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)