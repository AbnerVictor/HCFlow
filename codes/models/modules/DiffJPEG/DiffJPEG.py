# Pytorch
import torch
import torch.nn as nn
# Local
from .compression import compress_jpeg
from .decompression import decompress_jpeg
from .utils import diff_round, quality_to_factor
from .differentiable_quantize import choose_rounding
# from .differentiable_quantize import no_quantize_function, fft_quantization
import numpy as np


class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=80, quality_range=3):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): max quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        # if differentiable is True or differentiable=="gradient_1":
        #     rounding = differentiable_quantize.apply
        # elif differentiable == 'no_quantize':
        #     rounding = no_quantize_function.apply
        # elif differentiable == 'fft_quantize':
        #     rounding = fft_quantization
        # elif differentiable == 'undiff_round':
        #     rounding = torch.round
        rounding = choose_rounding(differentiable)
        # factor = quality_to_factor(quality)
        self.min_quality = quality-quality_range
        self.max_quality = quality+quality_range+1
        self.compress = compress_jpeg(rounding=rounding)
        self.decompress = decompress_jpeg()
        self.compress.requires_grad_(False)  # We fix the parameter of compression
        self.decompress.requires_grad_(True) # We update the parameter of decompression
        
    def forward(self, x, input_format='rgb', inter=False):
        if input_format != 'yuv422':
            _, _, h, w = x.shape
        else:
            _, h, w = x[0].shape
        # quality  min: self.min_quality   max: self.max_quality
        quality = np.random.randint(self.min_quality, self.max_quality)
        factor = quality_to_factor(quality)
        comp_new, comp_before = self.compress(x, factor, input_format)
        #y, cb, cr = self.compress(x,factor)
        # [1, 196, 8, 8], [1, 49, 8, 8]
        # x = tensor([[[[0.4824, 0.4706, 0.4784,  
        # y = tensor([[[[ 1.0800e+02,  2.1000e+01,  9.0000e+00
        y, cb, cr = comp_new['y'], comp_new['cb'], comp_new['cr'], 
        # recovered = tensor([[[[0.4766, 0.4680, 0.4622
        recovered = self.decompress(y, cb, cr, factor, h, w, inter=inter)
        return recovered
