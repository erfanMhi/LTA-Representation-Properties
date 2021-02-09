import os
import numpy as np
import pickle


class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:

            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

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


class ReplayFactory:
    @classmethod
    def get_replay_fn(cls, cfg):
        if cfg.replay_with_len:
            return lambda: ReplayWithLen(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)
        else:
            return lambda: Replay(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)
