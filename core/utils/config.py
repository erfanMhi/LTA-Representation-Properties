import os

from core.utils import torch_utils


class Config:
    def __init__(self):
        self.exp_name = 'test'
        self.data_root = None
        self.device = None
        self.run = 0
        self.param_setting = 0

        self.env_name = None
        self.state_dim = None
        self.action_dim = None
        self.max_steps = 0

        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.num_eval_episodes = 5
        self.timeout = None
        self.stats_queue_size = 10

        self.__env_fn = None
        self.logger = None

        self.tensorboard_logs = False
        self.tensorboard_interval = 100

        self.converge_window = 10
        self.converge_threshold = 1e-04
        self.linear_hidden_units = []
        self.coord_dim = 2

        # TODO: move normalizes to configs
        self.state_normalizer = None
        self.state_norm_coef = 1.0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0

        # self.eval_set_size = 1000
        # self.retain_tasks = 1
        self.replay_with_len = False

    def get_log_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        torch_utils.ensure_dir(d)
        return d

    def log_config(self):
        attrs = self.get_print_attrs()
        for param, value in attrs.items():
            self.logger.info('{}: {}'.format(param, value))

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        return attrs

    @property
    def env_fn(self):
        return self.__env_fn

    @env_fn.setter
    def env_fn(self, env_fn):
        self.__env_fn = env_fn
        self.state_dim = env_fn().state_dim
        self.action_dim = env_fn().action_dim

    def get_visualization_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting), "visualizations")
        torch_utils.ensure_dir(d)
        return d

    def get_parameters_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "parameters")
        torch_utils.ensure_dir(d)
        return d


class DQNAgentConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'DQNAgent'
        self.learning_rate = 0

        self.decay_epsilon = False
        self.epsilon = 0.1
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

        self.discount = None

        self.network_type = 'flat'
        self.batch_size = None
        self.use_target_network = True
        self.memory_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None

        self.val_net = None
        self.target_network_update_freq = None

        self.replay_fn = None

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.vf_constraint = None
        self.vf_constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"


        self.evaluate_lipschitz = False
        self.evaluate_distance = False
        self.evaluate_orthogonality = False
        self.evaluate_interference = False
        # self.evaluate_decorrelation = False
        self.evaluate_diversity = False
        self.evaluate_sparsity = False
        self.evaluate_regression = False
        self.save_params = False
        self.save_early = None
        self.visualize = False

        self.activation_config = {'name': 'None'}
        self.online_property = False

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'vf_constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn']:
            del attrs[k]
        return attrs

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run",
                            "{}_param_setting".format(self.param_setting))


class LaplaceConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'Laplace'

        self.replay = True
        self.memory_size = 50000
        self.replay_fn = None

        self.optimizer_type = "Adam"
        self.optimizer_fn = None
        self.learning_rate = 0.001
        self.batch_size = 128

        self.rep_config = None
        self.rep_fn = None
        self.lmbda = 0.9
        self.beta = 5.0
        self.delta = 0.05

    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer',
                  'logger', '_Config__env_fn', 'data_root',
                  'optimizer_fn', 'replay_fn', 'rep_fn']:
            del attrs[k]
        return attrs


class DQNRepAgentConfig(DQNAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'DQNRepAgent'
        self.rep_fn = None
        self.rep_config  = None
        self.goal_id = 0


class DQNAuxAgentConfig(DQNAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'DQNAuxAgent'
        self.visualize_aux_distance = False

    def get_print_attrs(self):
        attrs = super().get_print_attrs()
        for k in ['aux_fns']:
            del attrs[k]
        return attrs

# class DQNAuxAgentKnowUsefulAreaConfig(DQNAuxAgentConfig):
#     def __init__(self):
#         super().__init__()
#         self.agent = 'DQNAuxAgentKnowUsefulArea'

# class BaselineConfig(Config):
#     def __init__(self):
#         super().__init__()
#         self.agent = 'Baseline'


class EvaluateConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'PropertyEvaluation'

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn', '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'optimizer_fn']:
            del attrs[k]
        return attrs
