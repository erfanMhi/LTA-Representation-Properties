import os
import numpy as np
import pickle
import random
import collections



def discounted_sampling(ranges, discount):
    """Draw samples from the discounted distribution over 0, ...., n - 1,
    where n is a range. The input ranges is a batch of such n`s.

    The discounted distribution is defined as
    p(y = i) = (1 - discount) * discount^i / (1 - discount^n).

    This function implement inverse sampling. We first draw
    seeds from uniform[0, 1) then pass them through the inverse cdf
    floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
    to get the samples.
    """
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges).astype(np.int64)
    else:
        samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds)
                / np.log(discount))
        samples = np.floor(samples).astype(np.int64)
    return samples


def uniform_sampling(ranges):
    return discounted_sampling(ranges, discount=1.0)


def overlap(l1, l2):
    if isinstance(l1[0], int):
        return l1[0] <= l2[1] and l1[1] >= l2[0]
    if isinstance(l1[0], tuple):
        for in1 in l1:
            for in2 in l2:
                if overlap(in1, in2):
                    return True
        return False
    else:
        ValueError('List values are not valid: {}, {}'.format(l1, l2))


class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.episode_start = 0
        self.episodes_idx = []
        self.overlap_start = 0
        self.pos = 0

    def num_episodes(self):
        return len(self.episodes_idx)

    def feed(self, experience, done=False):
        if self.pos >= len(self.data):
            self.data.append(experience)
            if done:
                if self.episode_start != self.pos and self.episode_start+1 != self.pos:
                    self.episodes_idx.append((self.episode_start, self.pos))

        else:
            self.data[self.pos] = experience

            if done:
                # print('ep idx: ', self.episodes_idx)
                # print('ep start: ', self.episode_start)
                # print('ep end: ', self.pos)
                found_first_overlap = False

                if self.pos < self.episode_start:
                    episode_intervals = [(self.episode_start, self.memory_size-1), (0, self.pos)]
                else:
                    episode_intervals = [(self.episode_start, self.pos)]

                remove_idxs = []
                for idx, ep in enumerate(self.episodes_idx):
                    if ep[0] > ep[1]:
                        prev_episode_intervals = [(ep[0], self.memory_size - 1), (0, ep[1])]
                    else:
                        prev_episode_intervals = [(ep[0],  ep[1])]
                    # print(episode_intervals, prev_episode_intervals)
                    if overlap(episode_intervals, prev_episode_intervals):
                        if not found_first_overlap:
                            self.overlap_start = idx

                        found_first_overlap = True
                        remove_idxs.append(idx)

                if found_first_overlap:
                    for r_idx in sorted(remove_idxs, reverse=True):
                        self.episodes_idx.pop(r_idx)
                else:
                    self.overlap_start += 1

                if self.episode_start != self.pos and self.episode_start+1 != self.pos:
                    self.episodes_idx.insert(self.overlap_start, (self.episode_start, self.pos))

        self.pos = (self.pos + 1) % self.memory_size

        if done:
            self.episode_start = self.pos

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))

        return batch_data

    def get_buffer(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sampled_data = [self.data[ind] for ind in range(len(self.data))]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def _get_episode_length(self, episode_range):
        if episode_range[0] > episode_range[1]:
            return  self.memory_size-episode_range[0] + episode_range[1]
        else:
            return  episode_range[1] - episode_range[0] + 1

    def sample_pairs(self, discount=0.0):
        episodes_idx = self._sample_episodes(self.batch_size)
        step_ranges = np.array(list(map(self._get_episode_length, episodes_idx)))
        # print('step_ranges: ', step_ranges)
        step1_indices = uniform_sampling(step_ranges - 1)
        intervals = discounted_sampling(step_ranges - step1_indices - 1, discount=discount) + 1
        # print('intervals: ', intervals)
        step2_indices = step1_indices + intervals
        s1 = []
        s2 = []

        for epi_idx, step1_idx, step2_idx in zip(episodes_idx, step1_indices, step2_indices):
            s1.append(self.data[(epi_idx[0] + step1_idx) % self.memory_size])
            s2.append(self.data[(epi_idx[0] + step2_idx) % self.memory_size])

        # if self.pos % 1000 == 0:
        #     print('Pairs info')
        #     print(epi_idx)
        #     print((epi_idx[0] + step1_idx) % self.memory_size)
        #     print((epi_idx[0] + step2_idx) % self.memory_size)
        #     print(step1_indices)
        #     print(step2_indices)
        s1 = list(map(lambda x: np.asarray(x), zip(*s1)))
        s2 = list(map(lambda x: np.asarray(x), zip(*s2)))
        return s1[0], s2[0] # just giving back episodes

    def _sample_episodes(self, batch_size):
        indices = np.random.randint(len(self.episodes_idx), size=batch_size)
        return [self.episodes_idx[i] for i in indices]

    def sample_array(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]

        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def persist_memory(self, dir):
        for k in range(len(self.data)):
            transition = self.data[k]
            with open(os.path.join(dir, str(k)), "wb") as f:
                pickle.dump(transition, f)

class ReplayWithLen:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.exp_len = np.zeros(self.memory_size)
        self.pos = 0

    def set_exp_shape(self, exp_shape):
        self.data = np.zeros([self.memory_size]+exp_shape)

    def feed(self, experience, length):
        self.data[self.pos % self.memory_size] = experience
        self.exp_len[self.pos % self.memory_size] = length
        self.pos += 1

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = np.random.randint(0, min(self.memory_size, self.pos), size=batch_size)
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_len = [self.exp_len[ind] for ind in sampled_indices]
        return np.array(sampled_data), np.array(sampled_len)

    def sample_array(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = np.random.randint(0, min(self.memory_size, self.pos), size=batch_size)
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_len = [self.exp_len[ind] for ind in sampled_indices]
        return sampled_data, sampled_len

    def size(self):
        return min(self.memory_size, self.pos)


class MemoryOptimizedReplayBuffer(object):
    def __init__(self, memory_size, batch_size, ul_batch_size=None, ul_delta_T=None, frame_history_len=1, is_image=True):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.is_image = is_image

        self.size = memory_size
        self.batch_size = batch_size
        self.replay_T = ul_batch_size + ul_delta_T
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size=None):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""

        if batch_size is None:
            batch_size = self.batch_size

#        print(batch_size)
#        print(self.num_in_buffer)
#        return batch_size + 1 <= self.num_in_buffer

        return bool(self.num_in_buffer)

    def _encode_sample(self, idxes):
        #print(idxes)
        #print(self.num_in_buffer)
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size=None):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        if batch_size is None:
            batch_size = self.batch_size
    
        assert self.can_sample(batch_size)
        #print('here: ')
        #print(self.num_in_buffer)
        #print(batch_size)
        idx_high = self.num_in_buffer-1 if self.num_in_buffer == 1 else self.num_in_buffer - 2
        idxes = self._sample_n_unique(lambda: random.randint(0, idx_high), batch_size)
        #print('idxes: ', idxes)
        return self._encode_sample(idxes)

    def sample_ul(self, replay_T=None):
        """Sample `batch_size` different transitions.
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        delta_t: int
            Examples withing these timesteps will be considered as positive 
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        if replay_T is None:
            replay_T = self.replay_T
        assert self.num_in_buffer - self.replay_T - 2 > 0
        #idxes = self._sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        idx_high = self.num_in_buffer-1 if self.num_in_buffer == 1 else self.num_in_buffer - 2
        starting_idx = random.randint(0, idx_high- self.replay_T)
        idxes = np.arange(starting_idx, starting_idx+self.replay_T)
        return self._encode_sample(idxes)


    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32 if not self.is_image else np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

    def feed(self, experience):
        state, action, reward, next_state, done = experience
        idx = self.store_frame(state)
        self.store_effect(idx, action, reward, bool(done))
    
    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    #def size(self):
    #    return self.next_idx

    #def empty(self):
    #    return not len(self.data)

   # def persist_memory(self, dir):
   #     for k in range(len(self.data)):
   #         transition = self.data[k]
   #         with open(os.path.join(dir, str(k)), "wb") as f:
   #             pickle.dump(transition, f)


    def _sample_n_unique(self, sampling_f, n):
        """Helper function. Given a function `sampling_f` that returns
        comparable objects, sample n such unique objects.
        """
        res = []
        while len(res) < n:
            candidate = sampling_f()
        #    if candidate not in res: #y=uniqueness is gone to keep the implementation the same
        #        res.append(candidate)
            res.append(candidate)
        return res



class ReplayFactory:
    @classmethod
    def get_replay_fn(cls, cfg):
        if cfg.replay_with_len:
            return lambda: ReplayWithLen(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)
        else:
            return lambda: Replay(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)
            #return lambda: MemoryOptimizedReplayBuffer(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)

    @classmethod
    def get_ul_replay_fn(cls, cfg):
        return lambda: MemoryOptimizedReplayBuffer(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size, ul_batch_size=cfg.ul_batch_size, ul_delta_T=cfg.ul_delta_T)