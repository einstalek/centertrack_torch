import torch

import torch.nn as nn
import torch.nn.functional as F

from args import Args

from blazepalm.mobilenet import MobileNetV2


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(BlazeBlock, self).__init__()
        self.use_pooling = (stride == 2)
        self.channel_pad = out_channels - in_channels

        if self.use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            padding = 0
        else:
            padding = 1

        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.use_pooling:
            conv_input = F.pad(x, [0, 1, 0, 1], "constant", 0)  # TODO: check
            x = self.pool(x)
        else:
            conv_input = x

        conv_out = self.depth_conv(conv_input)
        conv_out = self.pointwise_conv(conv_out)

        if self.channel_pad > 0:
            x = F.pad(x, [0, 0, 0, 0, 0, self.channel_pad], "constant", 0)

        return self.relu(conv_out + x)


class BlazeUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlazeUpsampleBlock, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU6(inplace=True)
        self.blaze_block = BlazeBlock(out_channels, out_channels)

    def forward(self, x, previous_scale_output):
        x = self.relu(self.transpose_conv(x))
        x = self.blaze_block(x + previous_scale_output)
        return x


class OutputBlock(nn.Module):
    def __init__(self, num_scales=3, in_channels=(128, 256, 256), out_channels=(2, 2, 6), num_outputs=1):
        super(OutputBlock, self).__init__()
        self.num_scales = num_scales
        self.num_outputs = num_outputs
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1) for i in range(num_scales)
        ])

    def forward(self, scale_inputs):
        bs = scale_inputs[0].size()[0]
        scale_outputs = []
        for i in range(self.num_scales):
            output = self.convs[i](scale_inputs[i])
            output = output.permute(0, 2, 3, 1)
            output = torch.reshape(output, (bs, -1, self.num_outputs))
            scale_outputs.append(output)
        return torch.cat(scale_outputs, dim=1)


class BlazePalm(nn.Module):
    def __init__(self, args):
        super(BlazePalm, self).__init__()

        self.args = args
        self.heads = args.heads
        inp_dim = args.inp_dim

        self.pre_features = nn.Sequential(
            nn.Conv2d(inp_dim, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True))

        last_channel = None
        if self.args.backbone == 'blazepalm':
            self.backbone = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
                nn.ReLU6(inplace=True),
                BlazeBlock(32, 32),
                BlazeBlock(32, 32),
                BlazeBlock(32, 32),
                BlazeBlock(32, 32),
                BlazeBlock(32, 32),
                BlazeBlock(32, 32),
                BlazeBlock(32, 32),
                BlazeBlock(32, 64, stride=2),
                BlazeBlock(64, 64),
                BlazeBlock(64, 64),
                BlazeBlock(64, 64),
                BlazeBlock(64, 64),
                BlazeBlock(64, 64),
                BlazeBlock(64, 64),
                BlazeBlock(64, 64),
                BlazeBlock(64, 128, stride=2),
                BlazeBlock(128, 128),
                BlazeBlock(128, 128),
                BlazeBlock(128, 128),
                BlazeBlock(128, 128),
                BlazeBlock(128, 128),
                BlazeBlock(128, 128),
                BlazeBlock(128, 128)
            )
            last_channel = 128
        elif self.args.backbone == 'mobilenet':
            last_channel = 128
            self.backbone = MobileNetV2(in_channels=32, last_channel=last_channel)

        self.post_back_1 = nn.Sequential(
            BlazeBlock(last_channel, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256)
        )

        self.post_back_2 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256)
        )

        if self.args.use_conv_transp:
            self.deconv_layers = self._make_deconv_layer(1,
                                                         [128, ],
                                                         [self.args.num_kernels, ])
        self.head_convs = nn.ModuleList()
        for (head, dim) in self.heads.items():
            self.head_convs.append(nn.Sequential(
                nn.Conv2d(128, dim, stride=1, kernel_size=3, padding=1)))

        self.upsample_block_1 = BlazeUpsampleBlock(256, 256)
        self.upsample_block_2 = BlazeUpsampleBlock(256, 128)

    def _get_deconv_cfg(self, deconv_kernel, index):
        padding, output_padding = None, None
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters)
        assert num_layers == len(num_kernels)
        inplanes = 128

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.args.use_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU6(inplace=True))
            inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.pre_features(inputs)

        if self.args.backbone == 'blazepalm':
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

        first_scale_features = self.backbone(x)
        second_scale_features = self.post_back_1(first_scale_features)
        third_scale_features = self.post_back_2(second_scale_features)

        first_upscale_outputs = self.upsample_block_1(third_scale_features, second_scale_features)
        second_upscale_outputs = self.upsample_block_2(first_upscale_outputs, first_scale_features)

        if self.args.use_conv_transp:
            x = self.deconv_layers(second_upscale_outputs)
        else:
            x = second_upscale_outputs

        out = []
        for head in self.head_convs:
            out.append(head(x))
        return out


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
