# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
from .utils import *


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]), requires_grad=False)
        self.matrix = nn.Parameter(torch.from_numpy(matrix), requires_grad=False)

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)  # B, H, W, C
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
    #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result



class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    def __init__(self, block_size=8):
        super(block_splitting, self).__init__()
        self.k = block_size

    def forward(self, image):
        batch_size, height, width = image.shape
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)
    

class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor =  nn.Parameter(torch.from_numpy(tensor).float(), requires_grad=False)
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float(), requires_grad=False)
        
    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, image, factor):
        image_before = image.float() / (self.y_table * factor)
        image_after = self.rounding(image_before)
        diff = torch.abs(image_after - image_before)
        return image_after, diff


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, image, factor):
        image_before = image.float() / (self.c_table * factor)
        image_after = self.rounding(image_before)
        diff = torch.abs(image_after - image_before)
        return image_after, diff


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    def __init__(self, rounding=torch.round):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l1_yuv444 = nn.Sequential(
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding)
        self.y_quantize = y_quantize(rounding=rounding)

    def forward(self, image, factor, input_format='rgb'):
        if input_format == 'rgb':
            y, cb, cr = self.l1(image * 255.)
        if input_format == 'yuv444':
            y, cb, cr = self.l1_yuv444(image.permute(0, 2, 3, 1) * 255.)
        elif input_format == 'yuv420':
            y, cb, cr = image[0] * 255., image[1] * 255., image[2] * 255.
        components = {'y': y, 'cb': cb, 'cr': cr}
        components_new = {}
        components_diff = {}
        for k in components.keys():
            comp = self.l2(components[k])
            # [1, 112, 112] -> [1, 196, 8, 8]
            # [1, 56, 56] -> [1, 49, 8, 8]
            if k in ('cb', 'cr'):
                comp_new, comp_diff = self.c_quantize(comp, factor)
            else:
                comp_new, comp_diff = self.y_quantize(comp, factor)
            
            components_new[k] = comp_new
            components_diff[k] = comp_diff

        return components_new, components_diff
