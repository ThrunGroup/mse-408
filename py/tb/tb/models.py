from torch import Tensor, nn


class MultiLayerPerceptron(nn.Module):
    """A simple multi-layer perceptron model."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_layers: int = 2,
        n_hidden_width: int = 256,
        hidden_activation: nn.Module = nn.LeakyReLU(),
    ):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden_width), hidden_activation]
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden_width, n_hidden_width), hidden_activation]
        layers += [nn.Linear(n_hidden_width, n_output), nn.Softmax()]
        self.m = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.m(x)
