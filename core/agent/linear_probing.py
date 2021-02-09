import os
import numpy as np
from core.utils.torch_utils import tensor
from core.utils import torch_utils
import torch

def linear_probing(agent, cfg):
    class LinearProbing(agent):
        def __init__(self, cfg):
            super().__init__(cfg)
            data = np.load(os.path.join(cfg.data_root, cfg.linearprob_path["train"]))
            self.replay = cfg.replay_fn()
            self.replay.memory_size = len(data)
            self.dataset = self.extract_info(data)

            self.linear_prob_task = []
            for task in cfg.linear_prob_task:
                task["linear_optimizer"] = cfg.optimizer_fn(task["linear_fn"].parameters())
                self.linear_prob_task.append(task)

            self.linear_max_steps = self.cfg.linear_max_steps

            self.total_loss = np.zeros(cfg.stats_queue_size)
            self.min_val_loss = np.inf
            self.validation_count = 0
            self.tasks = {}#[i["task"] for i in self.cfg.linearprob_tasks]
            for i in self.cfg.linearprob_tasks:
                self.tasks[i["task"]] = {k:i[k] for k in i.keys()}

        def extract_info(self, dataset):
            for state in dataset:
                ki = self.env.get_useful(state)
                self.replay.feed([state, ki])

        def _coord_tensor(self, obs):
            return tensor(obs/self.env.max_x*2-1, self.cfg.device)

        def step(self):
            self.total_steps += 1
            self.ep_steps += 1
            self.learn_model()

        def learn_model(self):
            states, info = self.replay.sample()
            total_loss, _ = self.get_loss(states, info)
            # if len(total_loss) == 1:
            #     self.linear_optimizer.zero_grad()
            #     total_loss[0].backward()
            #     self.linear_optimizer.step()
            #     total = total_loss[0].item()
            # elif len(total_loss) == 2:
            #     self.linear_optimizer.zero_grad()
            #     total_loss[0].backward()
            #     self.linear_optimizer.step()
            #     total = total_loss[0].item()
            #     self.linear_optimizer_clf.zero_grad()
            #     total_loss[1].backward()
            #     self.linear_optimizer_clf.step()
            #     total = total_loss[0].item() + total_loss[1].item()
            # else:
            #     raise NotImplementedError
            total = 0
            for task in self.linear_prob_task:
                task["linear_optimizer"].zero_grad()
                total_loss[task["task"]].backward()
                task["linear_optimizer"].step()
                total += total_loss[task["task"]].item()
            self.update_loss_stats(total)

        def get_loss(self, states, info):
            with torch.no_grad():
                states = self.rep_net(self.cfg.state_normalizer(states))
            truth_dict = {}
            for t in self.tasks.keys():
                target = info[:, self.tasks[t]["truth_start"]: self.tasks[t]["truth_end"]]
                if t == "xy":
                    target = self._coord_tensor(target)
                elif t == "color":
                    target = target.reshape((-1))
                    target = torch_utils.tensor(target, self.cfg.device).type(torch.LongTensor)
                elif t == "count":
                    target = torch_utils.tensor(target, self.cfg.device)
                truth_dict[t] = target
            # info_xy = self._coord_tensor(info[:, :2])
            # info_color = info[:, 2:3].reshape((-1))
            # info_color = torch_utils.tensor(info_color, self.cfg.device).type(torch.LongTensor)
            # info_count = info[:, 14:17]
            # info_count = torch_utils.tensor(info_count, self.cfg.device)
            # truth_dict = {
            #     "xy": info_xy,
            #     "color": info_color,
            #     "count": info_count
            # }
            loss = {}
            percentage = {}
            for task in self.linear_prob_task:
                pred_info = task["linear_fn"](states)
                truth = truth_dict[task["task"]]
                l = task["loss_fn"](pred_info, truth)
                loss[task["task"]] = l
                pred_info = pred_info.detach().numpy()
                truth = truth.detach().numpy()
                # if task["task"] != "color":
                #     # percentage[task["task"]] = 1 - (np.abs(pred_info - truth).mean(axis=0) / (truth.max(axis=0)-truth.min(axis=0))).mean()
                #     percentage[task["task"]] = 1 - np.abs(pred_info - truth) / (1+np.abs(truth))
                # else:
                #     target = np.zeros((len(pred_info), task["num_class"]))
                #     target[:, truth] = 1
                #     percentage[task["task"]] = 1 - np.abs(pred_info - target).mean()
                if task["task"] == "color":
                    # target = np.zeros((len(pred_info), task["num_class"]))
                    # target[list(range(len(truth))), truth] = 1
                    # truth = target
                    # percentage[task["task"]] = 1 - np.abs(pred_info - truth)
                    pred_info = np.argmax(pred_info, axis=1)
                    percentage[task["task"]] = np.sum(pred_info == truth) / len(truth)
                    # percentage[task["task"]] = pred_info == truth
                else:
                    percentage[task["task"]] = 1 - np.abs(pred_info - truth) / (1 + np.abs(truth))
            return loss, percentage

        def update_loss_stats(self, total_loss):
            stats_idx = self.stats_counter % self.cfg.stats_queue_size
            self.total_loss[stats_idx] = total_loss
            self.stats_counter += 1

        def eval_linear_probing(self, states, info, validate=False):
            with torch.no_grad():
                loss_dir, percentage = self.get_loss(states, info)
            loss = 0
            perc = []
            for k in loss_dir.keys():
                loss += loss_dir[k].item()
                perc.append(percentage[k])
            if validate:
                return self.check_end(loss)
            else:
                return self.write_loss(loss, np.array(perc).mean())

        def check_end(self, loss):
            self.validation_count += 1
            if self.min_val_loss > loss:
                self.min_val_loss = loss
                self.count_stable = 0
            else:
                self.count_stable += 1

            log_str = 'total steps %d, total episodes %3d, ' \
                      'loss %.10f/%.10f (validate/minimum)'
            self.cfg.logger.info(log_str % (self.total_steps, self.num_episodes,
                                            loss, self.min_val_loss))

            if self.count_stable >= self.cfg.converge_window:
                return True
            else:
                return False

        def write_loss(self, loss, percentage_err):
            task = ""
            for i in self.cfg.linearprob_tasks:
                task += "_{}".format(i["task"])
            path = os.path.join(self.cfg.get_parameters_dir(), "../linear_probing{}.txt".format(task))
            with open(path, "r") as f:
                content = f.readlines()
            with open(path, "w") as f:
                f.write(content[0])
                f.write("Test set loss {:.8f}\n".format(loss))
                f.write("Percentage error {:.8f}\n".format(percentage_err))
            # print(percentage_err)

        def save_linear_probing(self):
            parameters_dir = self.cfg.get_parameters_dir()
            for task in self.linear_prob_task:
                path = os.path.join(parameters_dir, "linear_probing_{}".format(task["task"]))
                torch.save(task["linear_fn"].state_dict(), path)

            log_dir = self.cfg.get_log_dir()
            path = os.path.join(log_dir, "linear_probing_{}.txt".format(task["task"]))
            with open(path, "w") as f:
                f.write("Step {}. Validation loss {:.8f}\n".format(
                    self.total_steps,
                    self.min_val_loss
                ))

        def load_linear_probing(self):
            parameters_dir = self.cfg.get_parameters_dir()
            for task in self.linear_prob_task:
                path = os.path.join(parameters_dir, "linear_probing_{}".format(task["task"]))
                task["linear_fn"].load_state_dict(torch.load(path))


# def linear_probing(agent, cfg):
#     class LinearProbing(agent):
#         def __init__(self, cfg):
#             super().__init__(cfg)
#             self.replay = cfg.replay_fn()
#             self.linear_fn = self.cfg.linear_fn()
#             self.linear_optimizer = cfg.optimizer_fn(self.linear_fn.parameters())
#             self.linear_max_steps = self.cfg.linear_max_steps
#             self.rep_net = cfg.rep_fn()
#             if cfg.rep_config['load_params']:
#                 path = os.path.join(cfg.data_root, cfg.rep_config['path'])
#                 self.rep_net.load_state_dict(torch.load(path))
#                 path = path[:-7] + "val_net"
#                 self.val_net.load_state_dict(torch.load(path))
#                 self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
#                 self.targets.val_net.load_state_dict(self.val_net.state_dict())
#
#             self.total_loss = np.zeros(cfg.stats_queue_size)
#             # self.validation_loss = np.zeros(cfg.converge_window)
#             self.min_val_loss = np.inf
#             self.validation_count = 0
#
#         def _coord_tensor(self, obs):
#             return tensor(obs/self.env.max_x*2-1, self.cfg.device)
#
#         def step(self):
#             if self.reset is True:
#                 self.state = self.env.reset()
#                 self.reset = False
#
#             if np.random.rand() < self.cfg.eps_schedule():
#                 action = np.random.randint(0, self.cfg.action_dim)
#             else:
#                 with torch.no_grad():
#                     phi = self.rep_net(self.cfg.state_normalizer(self.state))
#                     q_values = self.val_net(phi)
#                 q_values = torch_utils.to_np(q_values).flatten()
#                 action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
#             next_state, _, done, _ = self.env.step([action])
#
#             self.feed_buffer(self.state, action, next_state, self.env.get_useful())
#
#             self.state = next_state
#             self.total_steps += 1
#             self.ep_steps += 1
#             self.learn_model()
#             if done or self.ep_steps == self.timeout:
#                 self.ep_steps = 0
#                 self.num_episodes += 1
#                 self.reset = True
#
#         def learn_model(self):
#             total_loss = self.get_loss()
#             self.linear_optimizer.zero_grad()
#             total_loss.backward()
#             self.linear_optimizer.step()
#             self.update_loss_stats(total_loss.item())
#
#         def feed_buffer(self, state, action, next_state, next_xy):
#             self.replay.feed([next_state, next_xy])
#
#         def get_loss(self):
#             states, coords = self.replay.sample()
#             with torch.no_grad():
#                 states = self.rep_net(self.cfg.state_normalizer(states))
#             coords = self._coord_tensor(coords)
#             pred_xy = self.linear_fn(states)
#             loss_s = torch.mean(torch.sum((pred_xy - coords)**2, 1))
#             return loss_s
#
#         def update_loss_stats(self, total_loss):
#             stats_idx = self.stats_counter % self.cfg.stats_queue_size
#             self.total_loss[stats_idx] = total_loss
#             self.stats_counter += 1
#
#         def eval_linear_probing(self, obs, truth, validate=False):
#             with torch.no_grad():
#                 rep = self.rep_net(self.cfg.state_normalizer(obs))
#             with torch.no_grad():
#                 prediction = self.linear_fn(rep)
#                 truth = self._coord_tensor(truth)
#                 loss = torch.mean(torch.sum((prediction - truth) ** 2, 1))
#
#             if validate:
#                 return self.check_end(loss)
#             else:
#                 return self.write_loss(loss)
#
#         def check_end(self, loss):
#             # self.validation_loss[self.validation_count % self.cfg.converge_window] = loss
#             self.validation_count += 1
#             # if np.max(self.validation_loss) - np.min(self.validation_loss) < self.cfg.converge_threshold:
#             #     return True
#             # else:
#             #     return False
#             if self.min_val_loss > loss:
#                 self.min_val_loss = loss
#                 self.count_stable = 0
#             else:
#                 self.count_stable += 1
#
#             log_str = 'total steps %d, total episodes %3d, ' \
#                       'loss %.10f/%.10f (validate/minimum)'
#             self.cfg.logger.info(log_str % (self.total_steps, self.num_episodes,
#                                             loss, self.min_val_loss))
#
#             if self.count_stable >= self.cfg.converge_window:
#                 return True
#             else:
#                 return False
#
#         def write_loss(self, loss):
#             path = os.path.join(self.cfg.get_parameters_dir(), "../linear_probing.txt")
#             with open(path, "a") as f:
#                 f.write("Test set loss {:.8f}\n".format(loss))
#
#         def save_linear_probing(self):
#             parameters_dir = self.cfg.get_parameters_dir()
#             path = os.path.join(parameters_dir, "linear_probing")
#             torch.save(self.linear_fn.state_dict(), path)
#             path = os.path.join(parameters_dir, "../linear_probing.txt")
#             with open(path, "w") as f:
#                 f.write("Step {}. Validation loss {:.8f}\n".format(
#                     self.total_steps,
#                     # self.validation_loss[(self.validation_count - 1) % self.cfg.converge_window]
#                     self.min_val_loss
#                 ))
#
#         def load_linear_probing(self):
#             parameters_dir = self.cfg.get_parameters_dir()
#             path = os.path.join(parameters_dir, "linear_probing")
#             self.linear_fn.load_state_dict(torch.load(path))
#
#
#     class LinearProbingNext(LinearProbing):
#         def __init__(self, cfg):
#             super().__init__(cfg)
#
#         def feed_buffer(self, state, action, next_state, next_xy):
#             # print(action, next_xy)
#             # import matplotlib.pyplot as plt
#             # plt.figure()
#             # plt.imshow(state)
#             # plt.show()
#             # exit()
#             self.replay.feed([state, action, next_xy])
#
#         def get_loss(self):
#             states, actions, coords = self.replay.sample()
#             with torch.no_grad():
#                 states = self.rep_net(self.cfg.state_normalizer(states))
#             states = states.reshape((len(states), -1))
#             if not isinstance(states, torch.Tensor): states = tensor(states, self.cfg.device)
#             lf_input = torch.cat((states, self.one_hot_tensor(actions)), 1)
#             coords = self._coord_tensor(coords)
#             pred_xy = self.linear_fn(lf_input)
#             loss_s = torch.mean(torch.sum((pred_xy - coords)**2, 1))
#             return loss_s
#
#         def one_hot_tensor(self, actions):
#             oha = np.zeros((len(actions), cfg.action_dim))
#             oha[np.arange(len(actions)), actions] = 1
#             return tensor(oha, self.cfg.device)
#
#         def eval_linear_probing(self, obs, truth, validate=False):
#             obs, actions = obs
#             actions = self.one_hot_tensor(actions)
#             with torch.no_grad():
#                 rep = self.rep_net(self.cfg.state_normalizer(obs))
#             if not isinstance(rep, torch.Tensor): rep = tensor(rep, self.cfg.device)
#             rep = rep.reshape((len(rep), -1))
#             with torch.no_grad():
#                 prediction = self.linear_fn(torch.cat((rep, actions), 1))
#                 truth = self._coord_tensor(truth)
#                 loss = torch.mean(torch.sum((prediction - truth) ** 2, 1))
#             if validate:
#                 return self.check_end(loss)
#             else:
#                 return self.write_loss(loss)

    if cfg.retain == "current":
        return LinearProbing(cfg)
    # elif cfg.retain == "next":
    #     return LinearProbingNext(cfg)
    else:
        raise NotImplementedError

