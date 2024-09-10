import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================== Channel Attention Map ===================================
# tìm ra channel nào đóng góp sự quan trọng nhất
# output sẽ là một refined feature map
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CAM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            Flatten(),  # (b x c)
            nn.Linear(in_channels, in_channels //
                      reduction_ratio),  # (b x (c/r))
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),  # (b x (c/r))
            nn.Linear(in_channels // reduction_ratio, in_channels),  # (b x c)
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.pool_types = pool_types

    def forward(self, x):
        # x: (b x c x h x w)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (b x c x 1 x 1)
                channel_att_raw = self.mlp(avg_pool) # (b x c)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))) # (b x c x 1 x 1)
                channel_att_raw = self.mlp(max_pool) # (b x c)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw # (b x c)
            else:
                channel_att_sum = channel_att_sum + channel_att_raw # (b x c)

        # print(channel_att_sum.unsqueeze(2).unsqueeze(3).shape) # (b x c x 1 x 1)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x) # (b x c x h x w)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], is_spatial=True):
        super(CBAM, self).__init__()
        self.cam = CAM(in_channels, reduction_ratio, pool_types)
        self.is_spatial = is_spatial
        if is_spatial:
            self.sam = SAM()

    def forward(self, x):
        out = self.cam(x)
        if self.is_spatial:
            out = self.sam(out)
        return out



# ================================== Spatial Attention Map ===================================
# tìm ra pixel nào đóng góp sự quan trọng nhất
# output sẽ là một refined feature map
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPooling(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.pool = ChannelPooling()
        self.conv = ConvLayer(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, relu=False)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        scale = torch.sigmoid(out)
        return scale * x
