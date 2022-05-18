import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_size, out_size, batch_size):
        super().__init__()

        self.linear_1 = nn.Linear(z_size, out_size**2)
        # the conv1d layer is use to create a whole batch of samples with every
        # forward pass of the Generator
        self.deconv_1 = nn.Conv1d(1, batch_size, out_size, stride=out_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = x[:, None]
        x = self.deconv_1(x)
        return x



class Critic(nn.Module):
    def __init__(self, out_size, batch_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(1,2),
            nn.Linear(out_size*batch_size, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
