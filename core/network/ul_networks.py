
import torch

from core.utils import torch_utils
from core.network import network_utils, network_bodies

class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
        init_type: initalization of the network layers 
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            init_type='default',
            device = 'cpu'
            ):
        super().__init__()

        
        if init_type == 'xavier':
            init_func = lambda layer: network_utils.layer_init_xavier(layer)
        elif init_type == 'default':
            init_func = lambda layer: layer
        else:
            raise ValueError('Init cannot be recognized')

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]

        sequence = list()
        for layer in hidden_layers:
            sequence.extend([init_func(layer), nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(init_func(torch.nn.Linear(last_size, output_size)))

        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)
        
        self.to(device)
        self.device = device
    
    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        if not isinstance(input, torch.Tensor):
            input = torch_utils.tensor(input, self.device)
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size

class UlEncoderModel(torch.nn.Module):

    def __init__(self, conv, latent_size, conv_out_size, device='cpu'):
        super().__init__()
        self.conv = conv  # Get from RL agent's model.
        self.head = torch.nn.Linear(conv_out_size, latent_size)

        self.to(device)
        self.device = device

    def forward(self, observation):
#        if observation.dtype == torch.uint8:
#            img = observation.type(torch.float)
#            img = img.mul_(1. / 255)
#        else:
#            img = observation
        if not isinstance(observation, torch.Tensor):
            observation = torch_utils.tensor(observation, self.device)
        conv = self.conv(observation)
        c = self.head(conv)

        return c


class ContrastModel(torch.nn.Module):

    def __init__(self, latent_size, anchor_hidden_sizes, device='cpu'):
        super().__init__()
        if anchor_hidden_sizes is not None:
            self.anchor_mlp = MlpModel(
                input_size=latent_size,
                hidden_sizes=anchor_hidden_sizes,
                output_size=latent_size,
            )
        else:
            self.anchor_mlp = None
        self.W = (torch.nn.Linear(latent_size, latent_size, bias=False))

        self.to(device)
        self.device = device
    
    def forward(self, anchor, positive):
        if not isinstance(anchor, torch.Tensor):
            anchor = torch_utils.tensor(anchor, self.device)
         
        if not isinstance(positive, torch.Tensor):
            anchor = torch_utils.tensor(positive, self.device)
 
        if self.anchor_mlp is not None:
            anchor = anchor + self.anchor_mlp(anchor)  # skip probably helps
        pred = self.W(anchor)
        logits = torch.matmul(pred, positive.T)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize
        return logits
