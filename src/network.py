import torch.nn as nn
import torch.nn.functional as F


class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=32 * 14 * 14, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=4)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        # Flatten.
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
