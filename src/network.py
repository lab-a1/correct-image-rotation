import torch.nn as nn


class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=12,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Dropout(0.25),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=12,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Dropout(0.25),
        )

        self.layer_3 = nn.Sequential(nn.Linear(in_features=512))
