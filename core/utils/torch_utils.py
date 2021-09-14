import os

import numpy as np
import torch


def select_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device('cuda')
    # if gpu_id >= 0:
    #     return torch.device('cuda:%d' % (gpu_id))
    else:
        return torch.device('cpu')


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x


def range_tensor(end, device):
    return torch.arange(end).long().to(device)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return True
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.12f} --> {val_loss:.12f}).  Saving model ...')
        model.store(training_step=model.total_steps)
        self.val_loss_min = val_loss


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
