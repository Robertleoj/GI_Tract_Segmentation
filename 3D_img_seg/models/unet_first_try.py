# Cell
import torch
import torch.nn as nn




class ConvBlockDown(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.conv = nn.Sequential(
            # Two convolutions that keep the same size
            nn.Conv3d(in_f, out_f, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(out_f, out_f, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlockUp(nn.Module):
    def __init__(self, n_concat, f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n_concat + f, f, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(f, f, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,nc, nf, n_classes, n_blocks):
        super().__init__()

        self.n_blocks = n_blocks


        # Make the convolutions in the downsampling stage
        self.conv_down = nn.ModuleList([ 
            ConvBlockDown(nc, nf)
        ])

        self.conv_down.extend([            
            ConvBlockDown(nf * (2 ** i), nf * (2 ** i)) for i in range(1, n_blocks)
        ])

        # Make the downsamplers
        self.downsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(nf * (2 ** i), nf * (2 ** (i + 1)), 4, 2, 1),
                nn.ReLU(),
                nn.BatchNorm3d(nf * (2 ** (i + 1))),
            ) for i in range(n_blocks - 1)
        ])


        # The convolutions in the upsampling
        self.conv_up = nn.ModuleList([ConvBlockUp(0, nf * (2 ** (n_blocks - 1)))])
        self.conv_up.extend([
            ConvBlockUp(nf * (2 ** i), nf * (2 ** i)) for i in range(n_blocks - 2, 0 - 1, -1)
        ])

        # The upsamplers
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(nf * (2 ** i), nf * (2 ** (i - 1)), 4, 2, 1),
                nn.ReLU(),
                nn.BatchNorm3d(nf * (2 ** (i - 1)))
            ) for i in range(n_blocks - 1, 0, -1)
        ])

        self.final_conv = nn.Conv3d(nf, n_classes, 3, 1, 1)

    def forward(self, x):
        down_outs = [] # keep for residual connections

        # Encode / downsampling
        for i in range(self.n_blocks - 1):
            x = self.conv_down[i](x)
            down_outs.append(x)
            x = self.downsamplers[i](x)

        # Pass through conv layers on the bottom
        x = self.conv_down[-1](x)
        x = self.conv_up[0](x)

        # Decode / upsampling
        for i in range(1, self.n_blocks):
            x = self.upsamplers[i - 1](x)
            down_out = down_outs.pop()
            catted = torch.cat([down_out, x], dim=1)
            x = self.conv_up[i](catted)

        return self.final_conv(x)

        

