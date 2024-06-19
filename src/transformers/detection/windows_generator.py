import torch
import torch.nn as nn


class WindowsGenerator(nn.Module):
    def __init__(self, windows=(8, 16, 32, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.windows = [[0, i - 1] for i in windows]
        self.windows = torch.as_tensor(self.windows, device='cuda')

    def forward(self, feature_maps, stride):
        windows = []
        batch_size = len(feature_maps)
        for i in range(batch_size):
            window = []
            feature_map = feature_maps[i]
            _, length = feature_map.shape
            for j in range(0, length, stride):
                window.append(self.windows + torch.as_tensor([j, j], device='cuda'))
            windows.append(window)
        windows = [torch.cat(window) for window in windows]
        return windows
