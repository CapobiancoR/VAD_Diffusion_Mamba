import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\nInput size of the image: {x.shape}")
        x = F.pixel_unshuffle(x, self.factor)
        print(f"Size after the pixel_unshuffle: {x.shape}")
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        print(f"Size after the view: {x.shape}")
        x = x.mean(dim=2)
        print(f"Size after the average: {x.shape}\n")
        return x
    
class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\nInput size of the downsampled: {x.shape}")
        x = x.repeat_interleave(self.repeats, dim=1)
        print(f"Size after interleave: {x.shape} (with repeats={self.repeats})")
        x = F.pixel_shuffle(x, self.factor)
        print(f"Size after pixel_shuffle: {x.shape}\n")

        return x    

images = torch.rand(1, 4, 224, 224)        

layer_down = PixelUnshuffleChannelAveragingDownSampleLayer(4, 8, 2)
layer_up = ChannelDuplicatingPixelUnshuffleUpSampleLayer(8, 4, 2)

# DownSample:
down_sampled = layer_down(images)

# Upsample:
up_sampled = layer_up(down_sampled)