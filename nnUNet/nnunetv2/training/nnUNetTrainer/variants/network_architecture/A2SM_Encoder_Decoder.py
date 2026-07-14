import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, \
    convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.torch_nn import act_layer
from timm.layers import DropPath
from einops import rearrange
import torch.nn.functional as F
from monai.networks.layers import DropPath, trunc_normal_


class OptInit:
    def __init__(self, drop_path_rate=0., pool_op_kernel_sizes_len=4):
        self.pool_op_kernel_sizes_len = pool_op_kernel_sizes_len
        self.conv = 'mr'
        self.act = 'leakyrelu'
        self.norm = 'instance'
        self.bias = True
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.use_stochastic = True
        self.drop_path = drop_path_rate
        # number of basic blocks in the backbone
        self.blocks = [1] * pool_op_kernel_sizes_len
        # number of reduce ratios in the backbone
        self.reduce_ratios = [16, 8, 4, 2] + [1] * (pool_op_kernel_sizes_len - 4)

class PatchEmbed(nn.Module):
    """
    Utilize a non-overlapping convolution to Down Sampling and Add Channels.
    """

    def __init__(self, in_channels, feature_size, patch_size, norm_layer=None):
        super(PatchEmbed,self).__init__()
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels=in_channels, out_channels=feature_size, kernel_size=3,
                              stride=1,padding=1)
        self.norm = norm_layer(feature_size) if norm_layer is not None else None

    def forward(self, x):
        _, _, D, H, W = x.size()

        # Calculate padding sizes once and apply
        pad_d = (self.patch_size[0] - D % self.patch_size[0]) if D % self.patch_size[0] != 0 else 0
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) if H % self.patch_size[1] != 0 else 0
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) if W % self.patch_size[2] != 0 else 0

        # Apply padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        # Apply convolution
        x = self.proj(x)

        # Apply normalization if needed
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)  # Flatten and transpose for norm layer
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.feature_size, D, Wh, Ww)

        return x

class CNNTrans(nn.Module):
    def __init__(self, output_channels,norm_layer):
        super(CNNTrans,self).__init__()
        self.output_channels = output_channels
        self.norm_layer = norm_layer
        self.pe=PatchEmbed(in_channels=output_channels, feature_size=output_channels, patch_size=(2, 1, 1),
                   norm_layer=self.norm_layer)
        self.vst=VSmixWindow_MSA(feature_size=output_channels, split_size=2, window_size=2,
                        num_head=24, img_size=[2, 3, 3], shift=True, qkv_bias=True,
                        attn_drop_rate=0.1, drop_rate=0.1)

        self.cnn=nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=3,padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.pe(x)
        x3 = self.vst(x2)
        return 0.5*(x1+x3)


# https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/residual_encoders.py
class A2SM_Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size: List[int],
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """
        :param input_channels:
        :param n_stages:
        :param features_per_stage: Note: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op:
        :param kernel_sizes:
        :param strides:
        :param n_blocks_per_stage:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block:
        :param bottleneck_channels: only needed if block is BottleneckD
        :param return_skips: set this to True if used as encoder in a U-Net like network
        :param disable_default_stem: If True then no stem will be created. You need to build your own and ensure it is executed first, see todo.
        The stem in this implementation does not so stride/pooling so building your own stem is a necessity if you need this.
        :param stem_channels: if None, features_per_stage[0] will be used for the default stem. Not recommended for BottleneckD
        :param pool_type: if conv, strided conv will be used. avg = average pooling, max = max pooling
        """

        super(A2SM_Encoder,self).__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        img_shape_list = []
        n_size_list = []
        pool_op_kernel_sizes = strides[1:]
        if conv_op == nn.Conv2d:
            h, w = patch_size[0], patch_size[1]
            img_shape_list.append((h, w))
            n_size_list.append(h * w)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                img_shape_list.append((h, w))
                n_size_list.append(h * w)

        elif conv_op == nn.Conv3d:
            h, w, d = patch_size[0], patch_size[1], patch_size[2]
            img_shape_list.append((h, w, d))
            n_size_list.append(h * w * d)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k, d_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                d //= d_k
                img_shape_list.append((h, w, d))
                n_size_list.append(h * w * d)

        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        img_min_shape = img_shape_list[-1]

        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        self.opt = opt
        self.opt.img_min_shape = img_min_shape
        self.n_swin_gnn_stages = 0  #n_swin_gnn_stages
        self.no_pool_gnn_stage_num = n_stages - 4
        self.n_conv_stages = self.no_pool_gnn_stage_num - self.n_swin_gnn_stages
        self.opt.n_size_list = n_size_list

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            if s < self.n_conv_stages:
                stage = nn.Sequential(
                    StackedResidualBlocks(n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s],
                                          kernel_sizes[s], stride_for_conv,
                                          conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                          nonlin_kwargs,
                                          block=block, bottleneck_channels=bottleneck_channels[s],
                                          stochastic_depth_p=stochastic_depth_p,
                                          squeeze_excitation=squeeze_excitation,
                                          squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio))
            elif s<n_stages-1:
                stage = nn.Sequential(
                    StackedResidualBlocks(n_blocks_per_stage[s] - 1, conv_op, input_channels, features_per_stage[s],
                                          kernel_sizes[s], stride_for_conv,
                                          conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                          nonlin_kwargs,
                                          block=block, bottleneck_channels=bottleneck_channels[s],
                                          stochastic_depth_p=stochastic_depth_p,
                                          squeeze_excitation=squeeze_excitation,
                                          squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio),
                    EViG(features_per_stage[s], img_shape_list[s], s - self.n_conv_stages, opt=self.opt,
                         conv_op=conv_op,
                         norm_op_kwargs=norm_op_kwargs)
                )
            else:
                output_channels = [features_per_stage[s]] * (n_blocks_per_stage[s] - 1)
                self.norm_layer = nn.LayerNorm
                stage = nn.Sequential(
                    block(conv_op, input_channels, output_channels[0], kernel_sizes[s], stride_for_conv, conv_bias,
                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                          stochastic_depth_p,
                          squeeze_excitation, squeeze_excitation_reduction_ratio),
                    # *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_sizes[s], 1, conv_bias, norm_op,
                    #         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                    #         squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, (n_blocks_per_stage[s] - 1))]
                    # PatchEmbed(in_channels=output_channels[0], feature_size=output_channels[0], patch_size=(2, 1, 1),
                    #            norm_layer=self.norm_layer),
                    # VSmixWindow_MSA(feature_size=output_channels[0], split_size=2, window_size=2,
                    #                 num_head=24, img_size=[2, 3, 3], shift=True, qkv_bias=True,
                    #                 attn_drop_rate=0.1, drop_rate=0.1)
                    CNNTrans(output_channels=output_channels[0],norm_layer=self.norm_layer)
                )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        # print("Encoder: ")
        for s_i in range(0, len(self.stages)):
            s = self.stages[s_i]
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class A2SM_Decoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder, A2SM_Encoder],
                 patch_size: List[int],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super(A2SM_Decoder,self).__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        img_shape_list = []
        n_size_list = []
        pool_op_kernel_sizes = strides[1:]
        if encoder.conv_op == nn.Conv2d:
            h, w = patch_size[0], patch_size[1]
            img_shape_list.append((h, w))
            n_size_list.append(h * w)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                img_shape_list.append((h, w))
                n_size_list.append(h * w)

        elif encoder.conv_op == nn.Conv3d:
            h, w, d = patch_size[0], patch_size[1], patch_size[2]
            img_shape_list.append((h, w, d))
            n_size_list.append(h * w * d)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k, d_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                d //= d_k
                img_shape_list.append((h, w, d))
                n_size_list.append(h * w * d)
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" % str(encoder.conv_op))

        img_min_shape = img_shape_list[-1]

        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        self.opt = opt
        self.opt.img_min_shape = img_min_shape
        self.n_swin_gnn_stages = 0  #n_swin_gnn_stages
        self.no_pool_gnn_stage_num = n_stages_encoder - 4
        self.n_conv_stages = self.no_pool_gnn_stage_num - self.n_swin_gnn_stages
        self.opt.n_size_list = n_size_list

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(nn.ConvTranspose3d(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))

            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if s < (n_stages_encoder - self.no_pool_gnn_stage_num):
                stages.append(nn.Sequential(
                    StackedResidualBlocks(n_conv_per_stage[s - 1] - 1, encoder.conv_op, 2 * input_features_skip,
                                          input_features_skip,
                                          encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op,
                                          encoder.norm_op_kwargs,
                                          encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin,
                                          encoder.nonlin_kwargs),

                    EViG(input_features_skip, img_shape_list[n_stages_encoder - (s + 1)],
                                  n_stages_encoder - self.no_pool_gnn_stage_num - (s + 1), opt=self.opt,
                                  conv_op=encoder.conv_op,
                                  norm_op_kwargs=encoder.norm_op_kwargs, dropout_op=encoder.dropout_op)
                )
                )

            else:
                stages.append(nn.Sequential(
                    StackedResidualBlocks(n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip,
                                          input_features_skip,
                                          encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op,
                                          encoder.norm_op_kwargs,
                                          encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin,
                                          encoder.nonlin_kwargs)
                ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.FM = FeatureMatch(32, 64)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        # print("Decoder: ")
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            if s < 4:
                x = torch.cat((x, skips[-(s + 2)]), 1)
            else:
                x = self.FM(x, skips[-(s + 2)])
            #x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0,
                 norm_op_kwargs=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(hidden_features, **norm_op_kwargs),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_features, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class MRConv(nn.Module):
    def __init__(self, in_channels, out_channels, offsets=[4]):
        super(MRConv, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.offsets = offsets
        self.mean = 0  # 用于存储 L1 差异的均值
        self.std = 0   # 用于存储 L1 差异的标准差

    def shift3d(self, x, dz, dy, dx):
        B, C, D, H, W = x.shape
        pad_d = abs(dz)
        pad_h = abs(dy)
        pad_w = abs(dx)
        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='replicate')
        d_start = pad_d + dz
        h_start = pad_h + dy
        w_start = pad_w + dx
        return x_pad[:, :, d_start:d_start+D, h_start:h_start+H, w_start:w_start+W]

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_j = torch.zeros_like(x)  # 初始化最大差异特征

        # 1. 计算全局统计量（均值和标准差）
        # 使用半尺寸位移作为参考（类似MRConv）
        x_rolled = torch.cat([x[:, :, -D//2:, :, :], x[:, :, :-D//2, :, :]], dim=2)
        x_rolled = torch.cat([x_rolled[:, :, :, -H//2:, :], x_rolled[:, :, :, :-H//2, :]], dim=3)
        x_rolled = torch.cat([x_rolled[:, :, :, :, -W//2:], x_rolled[:, :, :, :, :-W//2]], dim=4)
        norm = torch.abs(x - x_rolled)  # L1差异
        self.mean = torch.mean(norm)    # 全局均值
        self.std = torch.std(norm)      # 全局标准差

        # 2. 遍历所有偏移量，计算位移差异
        for dz in self.offsets:
            # 深度方向位移
            shifted_neg_d = self.shift3d(x, -dz, 0, 0)
            shifted_pos_d = self.shift3d(x, dz, 0, 0)
            diff_neg = torch.abs(shifted_neg_d - x)  # L1差异
            diff_pos = torch.abs(shifted_pos_d - x)
            # 自适应掩码：仅保留显著差异（小于 mean - std 的部分）
            mask_neg = (diff_neg < self.mean - self.std).float()
            mask_pos = (diff_pos < self.mean - self.std).float()
            x_j = torch.max(x_j, (shifted_neg_d - x) * mask_neg)  # 更新最大差异
            x_j = torch.max(x_j, (shifted_pos_d - x) * mask_pos)

        for dy in self.offsets:
            # 高度方向位移
            shifted_neg_h = self.shift3d(x, 0, -dy, 0)
            shifted_pos_h = self.shift3d(x, 0, dy, 0)
            diff_neg = torch.abs(shifted_neg_h - x)
            diff_pos = torch.abs(shifted_pos_h - x)
            mask_neg = (diff_neg < self.mean - self.std).float()
            mask_pos = (diff_pos < self.mean - self.std).float()
            x_j = torch.max(x_j, (shifted_neg_h - x) * mask_neg)
            x_j = torch.max(x_j, (shifted_pos_h - x) * mask_pos)

        for dx in self.offsets:
            # 宽度方向位移
            shifted_neg_w = self.shift3d(x, 0, 0, -dx)
            shifted_pos_w = self.shift3d(x, 0, 0, dx)
            diff_neg = torch.abs(shifted_neg_w - x)
            diff_pos = torch.abs(shifted_pos_w - x)
            mask_neg = (diff_neg < self.mean - self.std).float()
            mask_pos = (diff_pos < self.mean - self.std).float()
            x_j = torch.max(x_j, (shifted_neg_w - x) * mask_neg)
            x_j = torch.max(x_j, (shifted_pos_w - x) * mask_pos)

        # 3. 拼接原始特征与最大差异特征
        x_cat = torch.cat([x, x_j], dim=1)
        return self.nn(x_cat)

class GraphConv(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, out, drop_path=0.2):
        super(GraphConv, self).__init__()
        self.channels = in_channels
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
        )
        self.graph_conv = MRConv(in_channels, in_channels * 2)
        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out, kernel_size=1)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        x = self.conv(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, S, H, W) or (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """

    if len(x.shape) == 4:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        windows = rearrange(x, 'b (h p1) (w p2) c -> (b h w) p1 p2 c',
                            p1=window_size[0], p2=window_size[1], c=C)
        windows = windows.permute(0, 3, 1, 2)

    elif len(x.shape) == 5:
        B, C, S, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        windows = rearrange(x, 'b (s p1) (h p2) (w p3) c -> (b s h w) p1 p2 p3 c',
                            p1=window_size[0], p2=window_size[1], p3=window_size[2], c=C)
        windows = windows.permute(0, 4, 1, 2, 3)
    else:
        raise NotImplementedError('len(x.shape) [%d] is equal to 4 or 5' % len(x.shape))

    return windows


def window_reverse(windows, window_size, size_tuple):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size, window_size)
        window_size (int): Window size
        S (int): Slice of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, S ,H, W)
    """
    if len(windows.shape) == 4:
        H, W = size_tuple
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        windows = windows.permute(0, 2, 3, 1)
        x = rearrange(windows, '(b h w) p1 p2 c -> b (h p1) (w p2) c',
                      p1=window_size[0], p2=window_size[1], b=B, h=H // window_size[0], w=W // window_size[1])
        x = x.permute(0, 3, 1, 2)

    elif len(windows.shape) == 5:
        S, H, W = size_tuple
        B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
        windows = windows.permute(0, 2, 3, 4, 1)
        x = rearrange(windows, '(b s h w) p1 p2 p3 c -> b (s p1) (h p2) (w p3) c',
                      p1=window_size[0], p2=window_size[1], p3=window_size[2], b=B,
                      s=S // window_size[0], h=H // window_size[1], w=W // window_size[2])
        x = x.permute(0, 4, 1, 2, 3)
    else:
        raise NotImplementedError('len(x.shape) [%d] is equal to 4 or 5' % len(windows.shape))

    return x


class SwinGrapher(nn.Module):
    """
    SwinGrapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, img_shape, drop_path=0.0,
                 conv_op=nn.Conv3d, norm_op_kwargs=None, window_size=[3, 6, 6], shift_size=[0, 0, 0]):
        super(SwinGrapher, self).__init__()
        self.channels = in_channels
        self.conv_op = conv_op
        self.img_shape = img_shape
        self.window_size = window_size
        self.shift_size = shift_size

        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels, **norm_op_kwargs),
        )
        self.graph_conv = GraphConv(in_channels, in_channels * 2)
        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
            size_tuple = (H, W)
            h, w = self.img_shape
            assert h == H and w == W, "input features has wrong size"
        elif self.conv_op == nn.Conv3d:
            B, C, S, H, W = x.shape
            size_tuple = (S, H, W)
            s, h, w = self.img_shape
            assert s == S and h == H and w == W, "input features has wrong size"
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        if max(self.shift_size) > 0 and self.conv_op == nn.Conv2d:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
        elif max(self.shift_size) > 0 and self.conv_op == nn.Conv3d:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(2, 3, 4))
        else:
            shifted_x = x

        # partition windows
        # nW*B, C, window_size, window_size, window_size
        x_windows = window_partition(shifted_x, self.window_size)

        x = self.fc1(x_windows)

        x = self.graph_conv(x)

        shifted_x = window_reverse(x, self.window_size, size_tuple)
        shifted_x = self.fc2(shifted_x)

        # reverse cyclic shift
        if max(self.shift_size) > 0 and self.conv_op == nn.Conv2d:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))
        elif max(self.shift_size) > 0 and self.conv_op == nn.Conv3d:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(2, 3, 4))
        else:
            x = shifted_x

        x = self.drop_path(x) + _tmp
        return x


class PoolGrapher(nn.Module):
    """
    PoolGrapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, img_shape, drop_path=0.0, norm_op_kwargs=None):
        super(PoolGrapher, self).__init__()
        self.channels = in_channels
        self.img_shape = img_shape

        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels, **norm_op_kwargs),
        )
        self.graph_conv = GraphConv(in_channels, in_channels * 2)
        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class SwinGNNBlocks(nn.Module):
    def __init__(self, channels, img_shape, index, opt=None, conv_op=nn.Conv3d, norm_op_kwargs=None, **kwargs):
        super(SwinGNNBlocks, self).__init__()

        blocks = []
        pool_op_kernel_sizes_len = opt.pool_op_kernel_sizes_len
        act = opt.act
        drop_path = opt.drop_path
        blocks_num_list = opt.blocks
        img_min_shape = opt.img_min_shape

        self.n_blocks = sum(blocks_num_list)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        sum_blocks = sum(blocks_num_list[0:index])
        idx_list = [(k + sum_blocks) for k in range(0, blocks_num_list[index])]

        if conv_op == nn.Conv2d:
            H_min, W_min = img_min_shape
            max_num = int(H_min * W_min // 2)
            k_candidate_list = [2, 4, 8, 16, 32]
            max_k = min(k_candidate_list, key=lambda x: abs(x - max_num))
            min_k = max_num // (2 * 2)
            if pool_op_kernel_sizes_len >= 5:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)] + [min(min_k * 16, max_k)] * (pool_op_kernel_sizes_len - 5)
            else:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)][0:pool_op_kernel_sizes_len]

            max_dilation = (H_min * W_min) // max(k_list)
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1]
        elif conv_op == nn.Conv3d:
            H_min, W_min, D_min = img_min_shape
            max_num = int(H_min * W_min * D_min // 3)
            k_candidate_list = [2, 4, 8, 16, 32]
            max_k = min(k_candidate_list, key=lambda x: abs(x - max_num))
            min_k = max_num // (2 * 2 * 2)
            if pool_op_kernel_sizes_len >= 5:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)] + [min(min_k * 16, max_k)] * (pool_op_kernel_sizes_len - 5)
            else:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)][0:pool_op_kernel_sizes_len]

            max_dilation = (H_min * W_min * D_min) // max(k_list)
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1] * window_size[2]
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)

        i = index
        for j in range(blocks_num_list[index]):
            idx = idx_list[j]
            if conv_op == nn.Conv2d:
                shift_size = [window_size[0] // 2, window_size[1] // 2]
            elif conv_op == nn.Conv3d:
                shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
            else:
                raise NotImplementedError('conv operation [%s] is not found' % conv_op)

            blocks.append(nn.Sequential(
                SwinGrapher(channels, img_shape, drop_path=dpr[idx], conv_op=conv_op,
                            norm_op_kwargs=norm_op_kwargs, window_size=window_size, shift_size=shift_size),
                FFN(channels, channels * 4, act=act, drop_path=dpr[idx], norm_op_kwargs=norm_op_kwargs)))

        blocks = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        x = self.blocks(x)
        return x


class PoolGNNBlocks(nn.Module):
    def __init__(self, channels, img_shape, index, opt=None, conv_op=nn.Conv3d, norm_op_kwargs=None, **kwargs):
        super(PoolGNNBlocks, self).__init__()

        blocks = []
        pool_op_kernel_sizes_len = opt.pool_op_kernel_sizes_len
        act = opt.act
        drop_path = opt.drop_path
        blocks_num_list = opt.blocks
        img_min_shape = opt.img_min_shape

        self.n_blocks = sum(blocks_num_list)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        sum_blocks = sum(blocks_num_list[0:index])
        idx_list = [(k + sum_blocks) for k in range(0, blocks_num_list[index])]

        if conv_op == nn.Conv2d:
            H_min, W_min = img_min_shape
            max_num = int(H_min * W_min // 2)
            k_candidate_list = [2, 4, 8, 16, 32]
            max_k = min(k_candidate_list, key=lambda x: abs(x - max_num))
            min_k = max_num // (2 * 2)
            if pool_op_kernel_sizes_len >= 5:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)] + [min(min_k * 16, max_k)] * (pool_op_kernel_sizes_len - 5)
            else:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)][0:pool_op_kernel_sizes_len]

            max_dilation = (H_min * W_min) // max(k_list)
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1]
        elif conv_op == nn.Conv3d:
            H_min, W_min, D_min = img_min_shape
            max_num = int(H_min * W_min * D_min // 3)
            k_candidate_list = [2, 4, 8, 16, 32]
            max_k = min(k_candidate_list, key=lambda x: abs(x - max_num))
            min_k = max_num // (2 * 2 * 2)
            if pool_op_kernel_sizes_len >= 5:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)] + [min(min_k * 16, max_k)] * (pool_op_kernel_sizes_len - 5)
            else:
                k_list = [min(min_k, max_k), min(min_k * 2, max_k), min(min_k * 2, max_k), min(min_k * 4, max_k),
                          min(min_k * 8, max_k)][0:pool_op_kernel_sizes_len]

            max_dilation = (H_min * W_min * D_min) // max(k_list)
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1] * window_size[2]
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)

        i = index
        for j in range(blocks_num_list[index]):
            idx = idx_list[j]
            blocks.append(nn.Sequential(
                PoolGrapher(channels, img_shape, drop_path=dpr[idx], norm_op_kwargs=norm_op_kwargs),
                FFN(channels, channels * 4, act=act, drop_path=dpr[idx], norm_op_kwargs=norm_op_kwargs)))

        blocks = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        x = self.blocks(x)
        return x


class EViG(nn.Module):
    def __init__(self, channels, img_shape, index, opt=None, conv_op=nn.Conv3d, norm_op_kwargs=None, **kwargs):
        super(EViG, self).__init__()
        self.PoolGnn = PoolGNNBlocks(channels, img_shape, index, opt=opt,
                                     conv_op=conv_op, norm_op_kwargs=norm_op_kwargs)
        self.SoolGnn = SwinGNNBlocks(channels, img_shape, index, opt=opt, conv_op=conv_op,
                               norm_op_kwargs=norm_op_kwargs)

    def forward(self, x):
        x_p = self.PoolGnn(x)
        x_s = self.SoolGnn(x)
        return 0.5 * (x_p + x_s)

def window_partition_trans(x, D_sp, H_sp, W_sp, num_heads=None, is_Mask=False):
    B, D, H, W, C = x.shape
    if is_Mask:
        x = x.reshape(B, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp, C).contiguous()
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, D_sp * H_sp * W_sp, C)
    else:
        x = x.reshape(B, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp, C // num_heads, num_heads).contiguous()
        x = x.permute(0, 1, 3, 5, 8, 2, 4, 6, 7).contiguous().view(-1, num_heads, D_sp * H_sp * W_sp, C // num_heads)
    return x

def window_reverse_trans(x, D_sp, H_sp, W_sp, D, H, W):
    _, _, C = x.shape
    x = x.view(-1, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, C).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(-1, D, H, W, C).contiguous()
    return x

def compute_mask_trans(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition_trans(img_mask, window_size[0], window_size[1], window_size[2], is_Mask=True)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class VariableShapeAttention(nn.Module):
    def __init__(self, feature_size, idx, split_size, window_size, num_head, img_size, shift=False, attn_drop_rate=0.):
        super(VariableShapeAttention, self).__init__()
        self.num_head = num_head
        self.init_window_size(idx, img_size, split_size, window_size)
        head_dim = 4 * feature_size // num_head
        self.scale = head_dim ** -0.5
        self.shift = shift
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

        mesh_args = torch.meshgrid.__kwdefaults__
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.D_sp - 1) * (2 * self.H_sp - 1) * (2 * self.W_sp - 1),
                num_head,
            )
        )
        coords_d = torch.arange(self.D_sp)
        coords_h = torch.arange(self.H_sp)
        coords_w = torch.arange(self.W_sp)
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.D_sp - 1
        relative_coords[:, :, 1] += self.H_sp - 1
        relative_coords[:, :, 2] += self.W_sp - 1
        relative_coords[:, :, 0] *= (2 * self.H_sp - 1) * (2 * self.W_sp - 1)
        relative_coords[:, :, 1] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.q_linear = nn.Linear(feature_size * 3 , feature_size * 3 )
        self.k_linear = nn.Linear(feature_size * 3 , feature_size * 3 )
        self.v_linear = nn.Linear(feature_size * 3 , feature_size * 3 )

    def init_window_size(self, idx, img_size, split_size, window_size):
        if idx == 0:
            self.D_sp, self.H_sp, self.W_sp = window_size if img_size[0] > window_size else img_size[0], \
                window_size if img_size[1] > window_size else img_size[1], \
                window_size if img_size[2] > window_size else img_size[2],
            self.D_sf, self.H_sf, self.W_sf = self.D_sp // 2 if img_size[0] > self.D_sp else 0, \
                self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                self.W_sp // 2 if img_size[2] > self.W_sp else 0
        elif idx == 1:
            self.D_sp, self.H_sp, self.W_sp = split_size if img_size[0] > split_size else img_size[0], \
                img_size[1], \
                split_size if img_size[2] > split_size else img_size[2]
            self.D_sf, self.H_sf, self.W_sf = self.D_sp // 2 if img_size[0] > self.D_sp else 0, \
                0, \
                self.W_sp // 2 if img_size[2] > self.W_sp else 0
        elif idx == 2:
            self.D_sp, self.H_sp, self.W_sp = split_size if img_size[0] > split_size else img_size[0], \
                split_size if img_size[1] > split_size else img_size[1], \
                img_size[2]
            self.D_sf, self.H_sf, self.W_sf = self.D_sp // 2 if img_size[0] > self.D_sp else 0, \
                self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                0
        elif idx == 3:
            self.D_sp, self.H_sp, self.W_sp = img_size[0], \
                split_size if img_size[1] > split_size else img_size[1], \
                split_size if img_size[2] > split_size else img_size[2]
            self.D_sf, self.H_sf, self.W_sf = 0, \
                self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                self.W_sp // 2 if img_size[2] > self.W_sp else 0

    def forward(self, qkv):
        B, D, H, W, C = qkv.shape
        pad_l = pad_t = pad_d0 = 0
        #print('x',self.D_sp, self.H_sp, self.W_sp)
        #print('2',self.D_sf, self.H_sf, self.W_sf)

        pad_d1 = (self.D_sp - D % self.D_sp) % self.D_sp
        pad_b = (self.H_sp - H % self.H_sp) % self.H_sp
        pad_r = (self.W_sp - W % self.W_sp) % self.W_sp
        qkv = F.pad(qkv, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = qkv.shape

        if self.shift:
            qkv = torch.roll(qkv, shifts=(-self.D_sf, -self.H_sf, -self.W_sf), dims=(1, 2, 3))

        # 分别处理Q、K、V的投影
        q_proj = self.q_linear(qkv)
        k_proj = self.k_linear(qkv)
        v_proj = self.v_linear(qkv)

        # 分别进行窗口分区
        q = window_partition_trans(q_proj, self.D_sp, self.H_sp, self.W_sp, self.num_head)
        k = window_partition_trans(k_proj, self.D_sp, self.H_sp, self.W_sp, self.num_head)
        v = window_partition_trans(v_proj, self.D_sp, self.H_sp, self.W_sp, self.num_head)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        n = self.D_sp * self.H_sp * self.W_sp
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.shift:
            mask = compute_mask_trans(dims=[Dp, Hp, Wp], window_size=(self.D_sp, self.H_sp, self.W_sp),
                                shift_size=(self.D_sf, self.H_sf, self.W_sf), device=qkv.device)
            nw = mask.shape[0]
            attn = attn.view(attn.shape[0] // nw, nw, self.num_head, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_head, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.D_sp * self.H_sp * self.W_sp, C).contiguous()
        x = window_reverse_trans(x, self.D_sp, self.H_sp, self.W_sp, Dp, Hp, Wp)
        #print('shift',(self.D_sf, self.H_sf, self.W_sf))
        if self.shift:
            x = torch.roll(x, shifts=(self.D_sf, self.H_sf, self.W_sf), dims=(1, 2, 3))

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x


class VSmixWindow_MSA(nn.Module):
    def __init__(self,
                 feature_size,
                 split_size,
                 window_size,
                 num_head,
                 img_size,
                 shift=False,
                 qkv_bias=False,
                 attn_drop_rate=0.0,
                 drop_rate=0.0):
        super(VSmixWindow_MSA, self).__init__()
        self.num_head = num_head
        self.qkv = nn.Linear(feature_size, feature_size * 3, bias=qkv_bias)
        self.act1 = nn.GELU()
        self.conv1 = nn.Linear(feature_size * 3, feature_size)
        self.norm1 = nn.LayerNorm(feature_size, eps=1e-6)
        self.dep_conv = nn.Conv3d(feature_size, feature_size, kernel_size=3,padding=1)
        self.norm2 = nn.BatchNorm3d(num_features=feature_size)
        self.act2 = nn.LeakyReLU()
        self.window_size = window_size

        self.attns = nn.ModuleList([
            VariableShapeAttention(
                feature_size=feature_size // 4,
                idx=i % 4,
                split_size=split_size,
                window_size=window_size,
                num_head=num_head,
                img_size=img_size,
                shift=shift,
                attn_drop_rate=attn_drop_rate
            )
            for i in range(4)])

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.drop = nn.Dropout(drop_rate)
        self.reset_parameters()

        #self.proj = nn.Linear(feature_size * 3, feature_size)
        self.Conv = nn.Conv3d(feature_size * 3, feature_size, kernel_size=1,stride=1)
        self.proj_drop = nn.Dropout(drop_rate)

    def reset_parameters(self):
        if self.rate1 is not None:
            self.rate1.data.fill_(0.5)
        if self.rate2 is not None:
            self.rate2.data.fill_(0.5)
        self.dep_conv.bias.data.fill_(0.0)

    def forward(self, x):
        qkv = x.permute(0, 2, 3, 4, 1).contiguous()
        qkv = self.qkv(qkv)
        B, D, H, W, C = qkv.shape
        #Conv
        #B, D, H, W, C
        # conv_x = self.conv1(self.act1(qkv))
        # conv_x = self.norm1(conv_x).permute(0, 4, 1, 2, 3)
        # conv_x = self.dep_conv(conv_x)
        # conv_x = self.act2(self.norm2(conv_x)).permute(0, 2, 3, 4, 1)
        #Transformer
        x1 = self.attns[0](qkv[:, :, :, :, :C // 4])
        x2 = self.attns[1](qkv[:, :, :, :, C // 4:C // 4 * 2])
        x3 = self.attns[2](qkv[:, :, :, :, C // 4 * 2:C // 4 * 3])
        x4 = self.attns[3](qkv[:, :, :, :, C // 4 * 3:])
        attn_x = torch.cat([x1, x2, x3, x4], dim=-1)
        # 考虑注意力通道问题
        x = attn_x.permute(0, 4, 1, 2, 3).contiguous()
        x= self.Conv(x)
        #x = self.rate1 * attn_x + self.rate2 * conv_x
        x = self.drop(x)
        #x=x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad3d(max_disp, 0)

    def forward_run(self, x_1, x_2):
        x_2 = self.padlayer(x_2)
        offsetx, offsety, offsetz = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')

        w, h, d = x_1.shape[2], x_1.shape[3], x_1.shape[4]
        x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx + w, dy:dy + h, dz:dz + d], 1, keepdim=True)
                           for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))], 1)
        return x_out

    def forward(self, x_1, x_2):
        x = self.forward_run(x_1, x_2)
        return x


class FeatureMatch(nn.Module):  # input shape: n, c, h, w, d
    """Correlation-aware multi-window (CMW) MLP block."""

    def __init__(self, in_channels, num_channels):
        super().__init__()

        self.Corr = Correlation(max_disp=1)
        self.Conv = nn.Conv3d(91, num_channels, kernel_size=1, stride=1)
        #self.LayerNorm = nn.LayerNorm()
        #self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.ema = EMA3D(64)

    def forward(self, x_1, x_2):
        x_corr = self.Corr(x_1, x_2)
        x = torch.cat([x_1, x_corr, x_2], dim=1)
        #print('x',x.shape)
        #x = self.LayerNorm(x)
        x = self.Conv(x)
        #x = self.leaky_relu(x)
        _shortcut = x
        x = self.ema(x)
        x = x+_shortcut

        return x


class EMA3D(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA3D, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, d, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, d, h, w)  # b*g,c//g,h,w

        x_d = self.pool_d(group_x)
        x_h = self.pool_h(group_x).permute(0, 1, 3, 2, 4)
        x_w = self.pool_w(group_x).permute(0, 1, 4, 3, 2)

        dhw = self.conv1x1(torch.cat([x_d, x_h, x_w], dim=2))

        x_d, x_h, x_w = torch.split(dhw, [d, h, w], dim=2)

        x1 = self.gn(
            group_x * x_d.sigmoid() * x_h.permute(0, 1, 3, 2, 4).sigmoid() * x_w.permute(0, 1, 4, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, dhw

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, dhw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, d, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)

class A2SM(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size: List[int],
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_blocks_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_blocks_per_stage_decoder, int):
            n_blocks_per_stage_decoder = [n_blocks_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_blocks_per_stage_decoder) == (
                    n_stages - 1), "n_blocks_per_stage_decoder must have one less entries " \
                                   f"as we have resolution stages. here: {n_stages} " \
                                   f"stages, so it should have {n_stages - 1} entries. " \
                                   f"n_blocks_per_stage_decoder: {n_blocks_per_stage_decoder}"
        self.encoder = A2SM_Encoder(input_channels, patch_size, n_stages, features_per_stage, conv_op, kernel_sizes,
                                      strides,
                                      n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                      dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True)

        self.decoder = A2SM_Decoder(self.encoder, patch_size, strides, num_classes, n_blocks_per_stage_decoder,
                                      deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        # for s in skips:
        #     print(s.shape)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


if __name__ == '__main__':
    data = torch.rand((2, 1, 64, 160, 160))
    kwargs = {
        'A2SM': {
            'conv_bias': True,
            'norm_op': nn.BatchNorm3d,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    conv_or_blocks_per_stage = {
        'n_blocks_per_stage': [2, 2, 2, 2, 2, 2],
        'n_blocks_per_stage_decoder': [2, 2, 2, 2, 2]
    }
    model = A2SM(
        input_channels=1,
        patch_size=[64, 160, 160],
        n_stages=6,
        features_per_stage=[min(32 * 2 ** i,
                                320) for i in range(6)],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        num_classes=15,
        deep_supervision=True,
        **conv_or_blocks_per_stage,
        **kwargs['A2SM']
    )
    y = model(data)
    print(y[0].shape)
