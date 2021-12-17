import random
import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, args, buffer_size, batch_size):
        self.replay_buffer = []
        self.max_size = buffer_size
        self.batch_size = batch_size

        self.args = args
        self.n_actions = 4
        self.n_agents = 3
        self.state_shape = args.state_shape
        self.obs_shape = 46
        self.size = args.buffer_size
        self.episode_limit = args.episode_length
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def push(self, state, logits, reward, next_state, done):
        transition_tuple = (state, logits, reward, next_state, done)
        if len(self.replay_buffer) >= self.max_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(transition_tuple)

    def get_batches(self):
        sample_batch = random.sample(self.replay_buffer, self.batch_size)

        state_batches = np.array([_[0] for _ in sample_batch])
        action_batches = np.array([_[1] for _ in sample_batch])
        reward_batches = np.array([_[2] for _ in sample_batch])
        next_state_batches = np.array([_[3] for _ in sample_batch])
        done_batches = np.array([_[4] for _ in sample_batch])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def __len__(self):
        return len(self.replay_buffer)

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
