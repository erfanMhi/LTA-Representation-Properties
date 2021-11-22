from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.network import network_utils


class FCBody(nn.Module):
    def __init__(self, device, input_dim, hidden_units=(64, 64), activation=functional.relu, init_type='xavier'):
        super().__init__()
        self.to(device)
        self.device = device
        dims = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([network_utils.layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        if init_type == "xavier":
            self.layers = nn.ModuleList([network_utils.layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        elif init_type == "lta":
            self.layers = nn.ModuleList([network_utils.layer_init_lta(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.activation = activation
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            # print(layer(x).min(), layer(x).max())
            x = self.activation(layer(x))
        return x

    def compute_lipschitz_upper(self):
        return [np.linalg.norm(layer.weight.detach().cpu().numpy(), ord=2) for layer in self.layers]

    # def middle_ly(self, x, idx):
    #     for layer in self.layers[: idx]:
    #         x = self.activation(layer(x))
    #     return x


class ConvBody(nn.Module):
    def __init__(self, device, state_dim, architecture):
        super().__init__()

        def size(size, kernel_size=3, stride=1, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        spatial_length, _, in_channels = state_dim
        num_units = None
        layers = nn.ModuleList()
        for layer_cfg in architecture['conv_layers']:
            layers.append(nn.Conv2d(layer_cfg["in"], layer_cfg["out"], layer_cfg["kernel"],
                                         layer_cfg["stride"], layer_cfg["pad"]))
            if not num_units:
                num_units = size(spatial_length, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
            else:
                num_units = size(num_units, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
        num_units = num_units ** 2 * architecture["conv_layers"][-1]["out"]

        self.feature_dim = num_units
        self.spatial_length = spatial_length
        self.in_channels = in_channels
        self.layers = layers
        self.to(device)
        self.device = device

    def forward(self, x):
        x = functional.relu(self.layers[0](self.shape_image(x)))
        for layer in self.layers[1:]:
            x = functional.relu(layer(x))
        # return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)

    def shape_image(self, x):
        return x.reshape(-1, self.spatial_length, self.spatial_length, self.in_channels).permute(0, 3, 1, 2)

    def compute_lipschitz_upper(self):
        return [-1.0 for _ in self.layers]
        # def flatten(x, start, end=None):
        #     if not end: end = len(x.size())
        #     reduce((lambda x, y: x * y), x.size()[start: end+1])
        #     return x.view(x.size()[:start] + (reduce((lambda x, y: x * y), x.size()[start : end+1]),) + x.size()[end+1:])
        #
        # def convmatrix2d(kernel, image_shape):
        #     # kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
        #     # image: (in_channels, image_height, image_width, ...)
        #     assert image_shape[0] == kernel.shape[1]
        #     assert len(image_shape[1:]) == len(kernel.shape[2:])
        #     result_dims = torch.tensor(image_shape[1:]) - torch.tensor(kernel.shape[2:]) + 1
        #     m = torch.zeros((
        #         kernel.shape[0],
        #         *result_dims,
        #         *image_shape
        #     ))
        #     for i in range(m.shape[1]):
        #         for j in range(m.shape[2]):
        #             m[:, i, j, :, i:i + kernel.shape[2], j:j + kernel.shape[3]] = kernel
        #     return flatten(flatten(m, 0, len(kernel.shape[2:])), 1)
        #
        # # This code only works for one layer
        # assert len(self.layers) == 1
        #
        # l = 1.0
        #
        # for layer in self.layers:
        #     l = l * np.linalg.norm(convmatrix2d(layer.weight, (3, 15, 15)).detach().numpy(), ord=2)
        # return l, None


class DeConvBody(nn.Module):
    def __init__(self, device, in_dim, deconv_architecture, state_dim, conv_architecture):
        super().__init__()
        # Computing the shape of the input
        def size(size, kernel_size=3, stride=1, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        spatial_length, _, in_channels = state_dim
        num_units = None
        for layer_cfg in conv_architecture['conv_layers']:
            if not num_units:
                num_units = size(spatial_length, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
            else:
                num_units = size(num_units, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
        self.input_shape = (conv_architecture["conv_layers"][-1]["out"], num_units, num_units)

        layers = nn.ModuleList()
        for layer_cfg in deconv_architecture['deconv_layers']:
            layers.append(nn.ConvTranspose2d(layer_cfg["in"], layer_cfg["out"], layer_cfg["kernel"],
                                             layer_cfg["stride"], layer_cfg["pad"],
                                             output_padding=layer_cfg["out_pad"]))

        self.layers = layers
        self.to(device)
        self.device = device

    def forward(self, x):
        x = self.shape_input(x)
        for layer in self.layers[:-1]:
            x = functional.relu(layer(x))
        x = self.layers[-1](x)
        return x.permute(0, 2, 3, 1)

    def shape_input(self, x):
        k, w, h = self.input_shape
        return x.reshape(-1, k, w, h)

    def shape_output(self, x):
        return x.permute(0, 2, 3, 1)

    def compute_lipschitz_upper(self):
        raise NotImplementedError




class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size


#
# class TwoLayerFCBodyWithAction(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
#         super(TwoLayerFCBodyWithAction, self).__init__()
#         hidden_size1, hidden_size2 = hidden_units
#         self.fc1 = layer_init_xavier(nn.Linear(state_dim, hidden_size1))
#         self.fc2 = layer_init_xavier(nn.Linear(hidden_size1 + action_dim, hidden_size2))
#         self.gate = gate
#         self.feature_dim = hidden_size2
#
#     def forward(self, x, action):
#         x = self.gate(self.fc1(x))
#         phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
#         return phi
#
# class OneLayerFCBodyWithAction(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
#         super(OneLayerFCBodyWithAction, self).__init__()
#         self.fc_s = layer_init_xavier(nn.Linear(state_dim, hidden_units))
#         self.fc_a = layer_init_xavier(nn.Linear(action_dim, hidden_units))
#         self.gate = gate
#         self.feature_dim = hidden_units * 2
#
#     def forward(self, x, action):
#         phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
#         return phi
#
# class DummyBody(nn.Module):
#     def __init__(self, state_dim):
#         super(DummyBody, self).__init__()
#         self.feature_dim = state_dim
#
#     def forward(self, x):
#         return x
#
# class NatureConvBody(nn.Module):
#     def __init__(self, in_channels=4):
#         super(NatureConvBody, self).__init__()
#         self.feature_dim = 512
#         self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
#         self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
#         self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
#         self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
#
#     def forward(self, x):
#         y = F.relu(self.conv1(x))
#         y = F.relu(self.conv2(y))
#         y = F.relu(self.conv3(y))
#         y = y.view(y.size(0), -1)
#         y = F.relu(self.fc4(y))
#         return y
#
# class DDPGConvBody(nn.Module):
#     def __init__(self, in_channels=4):
#         super(DDPGConvBody, self).__init__()
#         self.feature_dim = 39 * 39 * 32
#         self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
#         self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))
#
#     def forward(self, x):
#         y = F.elu(self.conv1(x))
#         y = F.elu(self.conv2(y))
#         y = y.view(y.size(0), -1)
#         return y




