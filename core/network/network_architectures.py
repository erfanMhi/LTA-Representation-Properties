import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.network import network_utils, network_bodies
from core.utils import torch_utils


class LinearNetwork(nn.Module):
    def __init__(self, device, input_units, output_units, init_type='xavier'):
        super().__init__()

        if init_type == 'xavier':
            self.fc_head = network_utils.layer_init_xavier(nn.Linear(input_units, output_units))
        elif init_type == 'lta':
            self.fc_head = network_utils.layer_init_lta(nn.Linear(input_units, output_units))
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type)) 

        self.to(device)
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.fc_head(x)
        #print('weights: ', self.fc_head.weight)
        # if self.fc_head.weight.grad is not None:
        #    print('weights more than 0: ', torch.max(self.fc_head.weight.grad))
        #print('weights grads: ', self.fc_head.weight.grad)
        return y

    def compute_lipschitz_upper(self):
        return [np.linalg.norm(self.fc_head.weight.detach().numpy(), ord=2)]

class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=None):
        super().__init__()
        body = network_bodies.FCBody(input_units, hidden_units=tuple(hidden_units))
        self.fc_head = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, output_units))
        self.to(device)

        self.device = device
        self.body = body
        self.head_activation = head_activation

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)

        phi = self.body(x)
        phi = self.fc_head(phi)
        if self.head_activation is not None:
            phi = self.head_activation(phi)
        return phi

    def compute_lipschitz_upper(self):
        lips = self.body.compute_lipschitz_upper()
        lips.append(np.linalg.norm(self.fc_head.weight.detach().numpy(), ord=2))
        return lips


class FCBody(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units):
        super().__init__()
        hiddens = list.copy(hidden_units)
        hiddens.append(output_units)
        body = network_bodies.FCBody(input_units, hidden_units=tuple(hiddens))
        self.to(device)

        self.device = device
        self.body = body

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        phi = self.body(x)
        return phi

    def compute_lipschitz_upper(self):
        lips = self.body.compute_lipschitz_upper()
        return lips


class ConvNetwork(nn.Module):
    def __init__(self, device, state_dim, output_units, architecture, head_activation=None):
        super().__init__()

        self.conv_body = network_bodies.ConvBody(device, state_dim, architecture)
        if "fc_layers" in architecture:
            hidden_units = list.copy(architecture["fc_layers"]["hidden_units"])
            self.fc_body = network_bodies.FCBody(self.conv_body.feature_dim, hidden_units=tuple(hidden_units))
            self.fc_head = network_utils.layer_init_xavier(nn.Linear(self.fc_body.feature_dim, output_units))
        else:
            self.fc_body = None
            self.fc_head = network_utils.layer_init_xavier(nn.Linear(self.conv_body.feature_dim, output_units))

        self.to(device)
        self.device = device
        self.head_activation = head_activation

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch_utils.tensor(x, self.device)
        phi = self.conv_body(x)
        if self.fc_body:
            phi = self.fc_body(phi)
        phi = self.fc_head(phi)
        if self.head_activation is not None:
            phi = self.head_activation(phi)
        return phi

    def compute_lipschitz_upper(self):
        lips_fc = self.fc_body.compute_lipschitz_upper()
        lips_conv = self.conv_body.compute_lipschitz_upper()
        return lips_conv + lips_fc + [np.linalg.norm(self.fc_head.weight.detach().numpy(), ord=2)]


class ConvBody(nn.Module):
    def __init__(self, device, state_dim, output_units, architecture):
        super().__init__()

        self.conv_body = network_bodies.ConvBody(device, state_dim, architecture)
        if "fc_layers" in architecture:
            hidden_units = list.copy(architecture["fc_layers"]["hidden_units"])
            hidden_units.append(output_units)
            self.fc_body = network_bodies.FCBody(self.conv_body.feature_dim, hidden_units=tuple(hidden_units))
        else:
            hidden_units = [output_units]
            self.fc_body = network_bodies.FCBody(self.conv_body.feature_dim, hidden_units=tuple(hidden_units))
        self.to(device)
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch_utils.tensor(x, self.device)
        phi = self.conv_body(x)
        phi = self.fc_body(phi)
        return phi

    def compute_lipschitz_upper(self):
        lips_fc = self.fc_body.compute_lipschitz_upper()
        lips_conv = self.conv_body.compute_lipschitz_upper()
        return lips_conv + lips_fc


class DeConvNetwork(nn.Module):
    def __init__(self, device, in_dim, out_dim, deconv_architecture, state_dim, conv_architecture):
        super().__init__()
        if "fc_layers" in deconv_architecture:
            hidden_units = list.copy(deconv_architecture["fc_layers"]["hidden_units"])
            self.fc_body = network_bodies.FCBody(in_dim, hidden_units=tuple(hidden_units))
            self.deconv_body = network_bodies.DeConvBody(device, self.fc_body.feature_dim,
                                                         deconv_architecture,
                                                         state_dim,
                                                         conv_architecture)
        else:
            self.fc_body = None
            self.deconv_body = network_bodies.DeConvBody(device, in_dim, deconv_architecture, state_dim, conv_architecture)
        self.to(device)
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch_utils.tensor(x, self.device)
        if self.fc_body:
            # phi = functional.relu(self.fc_body(x))
            phi = self.fc_body(x)
        else:
            phi = x
        phi = self.deconv_body(phi)
        return phi

    def get_last_layer(self, x):
        raise NotImplementedError

    def compute_lipschitz_upper(self):
        raise NotImplementedError


class LinearActionNetwork(nn.Module):
    def __init__(self, device, input_units, output_units, num_actions):
        super().__init__()
        self.fc_head = network_utils.layer_init_xavier(nn.Linear(input_units, output_units * num_actions))
        self.to(device)
        self.device = device
        self.output_units = output_units
        self.num_actions = num_actions

    def forward(self, x, a):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.fc_head(x)
        y = y.reshape(-1, self.num_actions, self.output_units)
        return y[torch.arange(y.size(0)).long(), a]

    def compute_lipschitz_upper(self):
        raise NotImplementedError


class FCActionNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, num_actions):
        super().__init__()
        body = network_bodies.FCBody(input_units, hidden_units=tuple(hidden_units))
        self.fc_head = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, output_units * num_actions))
        self.to(device)
        self.output_units = output_units
        self.num_actions = num_actions
        self.device = device
        self.body = body

    def forward(self, x, a):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)

        phi = self.body(x)
        y = self.fc_head(phi)
        y = y.reshape(-1, self.num_actions, self.output_units)
        return y[torch.arange(y.size(0)).long(), a]

    def compute_lipschitz_upper(self):
        lips = self.body.compute_lipschitz_upper()
        lips.append(np.linalg.norm(self.fc_head.weight.detach().numpy(), ord=2))
        return lips
