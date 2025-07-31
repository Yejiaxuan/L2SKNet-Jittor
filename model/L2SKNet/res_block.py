import jittor as jt
import jittor.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def execute(self, x):
        residual = x
        x = self.body(x)

        if self.shortcut:
            residual = self.shortcut(residual)

        out = jt.nn.relu(x + residual)
        return out
