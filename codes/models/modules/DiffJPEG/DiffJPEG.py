# Pytorch
import torch
import torch.nn as nn
# Local
from models.modules.DiffJPEG.compression import compress_jpeg, block_splitting
from models.modules.DiffJPEG.decompression import decompress_jpeg, block_merging
from models.modules.DiffJPEG.utils import diff_round, quality_to_factor
from models.modules.DiffJPEG.differentiable_quantize import choose_rounding
# from .differentiable_quantize import no_quantize_function, fft_quantization
import numpy as np


class DiffJPEG(nn.Module):
    def __init__(self, differentiable='fft_quantize', quality=80, quality_range=3):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): max quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        rounding = choose_rounding(differentiable)
        self.min_quality = quality - quality_range
        self.max_quality = quality + quality_range + 1
        self.compressor = compress_jpeg(rounding=rounding)
        self.decompressor = decompress_jpeg(requires_grad=False)
        self.block_merge = block_merging()
        self.block_split = block_splitting()

        self.compressor.requires_grad_(False)  # We fix the parameter of compression
        self.decompressor.requires_grad_(False)  # We update the parameter of decompression

    def forward(self, x, input_format='rgb', inter=False, save_jpg_path=None):
        if input_format != 'yuv420':
            _, _, h, w = x.shape
        else:
            _, h, w = x[0].shape
        # quality  min: self.min_quality   max: self.max_quality
        quality = np.random.randint(self.min_quality, self.max_quality)
        factor = quality_to_factor(quality)
        comp_new, comp_before = self.compressor(x, factor, input_format)
        # y, cb, cr = self.compress(x,factor)
        # [1, 196, 8, 8], [1, 49, 8, 8]
        # x = tensor([[[[0.4824, 0.4706, 0.4784,  
        # y = tensor([[[[ 1.0800e+02,  2.1000e+01,  9.0000e+00
        y, cb, cr = comp_new['y'], comp_new['cb'], comp_new['cr'],
        if save_jpg_path is not None:
            pil_jpeg_readout, size_kb = save_jpg_torchjpeg(y, cb, cr, 
                            self.compressor.y_quantize.y_table, 
                            self.compressor.c_quantize.c_table,
                            factor, h, w,
                            out_jpg_path=save_jpg_path)
        else:
            pil_jpeg_readout, size_kb = None, None
        # recovered = tensor([[[[0.4766, 0.4680, 0.4622
        recovered = self.decompressor(y, cb, cr, factor, h, w, inter=inter)
        return recovered

def save_jpg_torchjpeg(y, cb, cr, q_y, q_c, factor, H, W, out_jpg_path='./save_from_npy.jpg'):

    def reshape_channel1(x, block=8):
        n, c, h, w = x.shape
        return torch.tensor(x.reshape(n, H//block, W//block, 8, 8))
    # dimensions, test_quantization, 
    test_Y_coefficients = reshape_channel1(y)
    Cb_coefficients = reshape_channel1(cb, block=16) # todo test 8 or 16
    Cr_coefficients = reshape_channel1(cr, block=16)
    test_dimensions = torch.tensor([[H, W],
                            [H//2, W//2],
                            [H//2, W//2]]).type(torch.IntTensor)
    # test_dimensions = (test_dimensions*8).type_as(dimensions)
    # test_dimensions = (test_dimensions*8)
    test_quantization = (torch.stack([q_y, q_c, q_c])*factor).type(torch.ShortTensor)
    test_Y_coefficients = test_Y_coefficients.type(torch.ShortTensor)
    test_CbCr_coefficients = torch.cat([Cb_coefficients, Cr_coefficients], dim=0).type(torch.ShortTensor)
    # dimensions
    # tensor([[480, 480],
    #         [240, 240],
    #         [240, 240]], dtype=torch.int32)
    # CbCr_coefficients
    # Y_coefficients.shape = torch.Size([1, 60, 60, 8, 8])
    # CbCr_coefficients.shape = torch.Size([2, 30, 30, 8, 8])
    # test_quantization.shape = torch.Size([3, 8, 8])
    import torchjpeg.codec
    from PIL import Image
    torchjpeg.codec.write_coefficients(out_jpg_path, test_dimensions, test_quantization, test_Y_coefficients, test_CbCr_coefficients)
    readout_dimensions, readout_quantization, readout_Y_coefficients, readout_CbCr_coefficients = torchjpeg.codec.read_coefficients(out_jpg_path)
    assert torch.equal(readout_dimensions, test_dimensions)
    assert torch.equal(readout_quantization, test_quantization)
    assert torch.equal(readout_Y_coefficients, test_Y_coefficients)
    assert torch.equal(readout_CbCr_coefficients, test_CbCr_coefficients)
    pil_jpeg_readout = np.array(Image.open(out_jpg_path))
    import os
    size_kb = os.path.getsize(out_jpg_path)/1024.0
    # print(size_kb)
    return pil_jpeg_readout, size_kb

class DiffJPEG_new(DiffJPEG):
    def __init__(self, differentiable='fft_quantize', quality=80, quality_range=3):
        super(DiffJPEG_new, self).__init__(differentiable, quality, quality_range)
        self.learnable_decompressor = decompress_jpeg(requires_grad=True)
        self.learnable_decompressor.requires_grad_(True)

    def compress(self, x, input_format='rgb', diff=False):
        if input_format != 'yuv420':
            _, _, h, w = x.shape
        else:
            _, h, w = x[0].shape

        # quality  min: self.min_quality   max: self.max_quality
        quality = np.random.randint(self.min_quality, self.max_quality)
        factor = quality_to_factor(quality)
        comp_new, comp_diff = self.compressor(x, factor, input_format)

        # Merge blocks
        comp_new['y'] = self.block_merge(comp_new['y'], h, w)
        comp_new['cb'] = self.block_merge(comp_new['cb'], h // 2, w // 2)
        comp_new['cr'] = self.block_merge(comp_new['cr'], h // 2, w // 2)

        if diff:
            return factor, comp_new, comp_diff
        else:
            return factor, comp_new

    def decompress(self, x, factor, mode='fix'):
        y, cb, cr = x['y'], x['cb'], x['cr']  # dct coefficient
        _, h, w = y.shape # B, H, W

        # Split to blocks
        y = self.block_split(y)  # B H//8 W//8 8 8
        cb = self.block_split(cb)
        cr = self.block_split(cr)

        if mode == 'fix':
            recovered = self.decompressor(y, cb, cr, factor, h, w)
            return recovered

        elif mode == 'learnable':
            recovered, inter = self.learnable_decompressor(y, cb, cr, factor, h, w, inter=True)
            return recovered, inter

    def forward(self, x, input_format='rgb', mode='fix'):
        factor_, compressed_x = self.compress(x, input_format)

        # 纯Encode过程应该试用不可学习参数的Compression Module
        recovered = self.decompress(compressed_x, factor_, mode='fix')

        if mode == 'fix':
            return recovered

        elif mode == 'train':
            # 训练过程中可以使用可学习参数的Decompress Module
            # 约等于，解码过程中Decoder可以使用可学习dequantization参数
            # _, inter_results = self.decompress(compressed_x, factor_, mode='learnable')
            # ycbcr = inter_results['merging']
            return recovered, factor_, compressed_x

        else:
            raise NotImplementedError

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt


    def tensor2np(x):
        return x.detach().cpu().numpy()


    im = cv2.imread('../../../datasets/demo_images/filtering/barbara.png') / 255.0

    plt.figure()
    plt.imshow(im)

    in_ten = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

    diff_jpeg = DiffJPEG_new(quality=10, quality_range=0)

    recovered = diff_jpeg(in_ten, input_format='rgb', mode='fix')

    plt.figure()
    plt.imshow(tensor2np(recovered.squeeze(0).permute(1, 2, 0))[:, :, ::-1])
    plt.show(block=True)
