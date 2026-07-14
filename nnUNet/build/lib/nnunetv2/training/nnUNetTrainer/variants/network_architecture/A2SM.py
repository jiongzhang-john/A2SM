from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn import BatchNorm3d
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.A2SM_Encoder_Decoder import A2SM_Encoder, A2SM_Decoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

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
        assert len(n_blocks_per_stage_decoder) == (n_stages - 1), "n_blocks_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_blocks_per_stage_decoder: {n_blocks_per_stage_decoder}"
        self.encoder = A2SM_Encoder(input_channels, patch_size, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True)

        self.decoder = A2SM_Decoder(self.encoder, patch_size, strides, num_classes, n_blocks_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

if __name__ == '__main__':
    data = torch.rand((2, 1, 64, 160, 160))
    kwargs = {
        'A2SM': {
            'conv_bias': True,
            'norm_op': BatchNorm3d,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    conv_or_blocks_per_stage = {
        'n_blocks_per_stage': [2,2,2,2,2,2],
        'n_blocks_per_stage_decoder': [2,2,2,2,2]
    }
    model = A2SM(
        input_channels=1,
        patch_size=[64,160,160],
        n_stages=6,
        features_per_stage=[min(32 * 2 ** i,
                                512) for i in range(6)],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3,3,3],[3,3,3],[3,3,3], [3,3,3], [3,3,3],[3,3,3]],
        strides=[[1,1,1],[1,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        num_classes=15,
        deep_supervision=True,
        **conv_or_blocks_per_stage,
        **kwargs['A2SM']
    )
    y=model(data)
    print(y[0].shape)
    
