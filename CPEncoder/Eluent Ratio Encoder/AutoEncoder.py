import torch
from torch import nn

class AE(nn.Module):

    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.to(device)
        #[b,54] -> [b,128]
        self.encoder = nn.Sequential(
            nn.Linear(50,50),
            # nn.Softplus(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(50,32),
            # nn.Softplus(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.Linear(64,20),
            # nn.ReLU()
        )

        #[b,128] ->  [b,54]
        self.decoder = nn.Sequential(
            nn.Linear(32,50),
            # nn.Softplus(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(50,50),
            # nn.Tanh()
            # nn.ReLU(),
            # nn.Linear(256,784),
            nn.Sigmoid()
        )

    def forward(self, x, device):
        """
        x:[B, feature = 54]
        B: batch size
        """

        batch_size = x.size(0)
        #ecnoder
        x = self.encoder(x)
        #decoder
        x = self.decoder(x)
        #reshape
        # x = x.view(batch_size,54)
        x.to(device)
        # print(f"forward_x_size: {x.size}")
        return x