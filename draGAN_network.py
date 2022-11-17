import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, out_size, batch_size, z_size):
        super().__init__()
        self.z_size = z_size
        self.out_size = out_size
        self.batch_size = batch_size


        self.linear = nn.Linear(z_size, out_size**2)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=batch_size,
            kernel_size=out_size,
            stride=out_size,
            groups=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        x = x[:, None]
        x = self.conv(x)
        return x


class Critic(nn.Module):
    def __init__(self, out_size, batch_size):
        super().__init__()
        self.flatten = nn.Flatten(1,2)
        self.linear_1 = nn.Linear(out_size*batch_size, 64)
        self.linear_2 = nn.Linear(64, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.linear_4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.batch_norm = nn.BatchNorm1d(num_features=64)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.batch_norm(x)

        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_3(x)
        x = self.leaky_relu(x)

        x = self.linear_4(x)

        return x
