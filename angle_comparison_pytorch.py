import torch
import torch.nn as nn


class AngleComparison(nn.Module):
    def __init__(self, filters=32, kernel_size=3):
        super(AngleComparison, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

    def forward(self, x):
        x1 = self.conv1(x.unsqueeze(1))
        x2 = self.conv2(x.unsqueeze(1))

        dot = torch.matmul(x1.transpose(1, 2), x2)
        norm1 = torch.norm(x1, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(x2, p=2, dim=1, keepdim=True)
        norm = torch.mul(norm1, norm2)

        angle = dot / norm
        output = torch.cat([angle, x], dim=1)

        return output
