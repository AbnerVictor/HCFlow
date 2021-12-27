import torch
import torch.nn as nn
import math

def choose_rounding(rounding_type):
    if rounding_type is True or rounding_type=="gradient_1":
        rounding = differentiable_quantize.apply
    elif rounding_type == 'no_quantize':
        rounding = no_quantize_function.apply
    elif rounding_type == 'fft_quantize':
        rounding = fft_quantization
    elif rounding_type == 'undiff_round':
        rounding = torch.round
    return rounding

class differentiable_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

def fft_quantization(input_tensor):
    test = 0
    for n in range(1, 10):
        test += math.pow(-1, n+1) / n * torch.sin(2 * math.pi * n * input_tensor)
    final_tensor = input_tensor - 1 / math.pi * test
    return final_tensor

class no_quantize_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
    
class DifferentiableQuantize(nn.Module):
    def __init__(self):
        super(DifferentiableQuantize, self).__init__()

    def forward(self, input):
        dq=differentiable_quantize.apply
        output = dq(input)
        return output


if __name__ == "__main__":
    a = torch.tensor([3.0,4.2,5,6,7,8,9,0])
    a.requires_grad_()
    dq=DifferentiableQuantize()
    b=dq(a)
    # b=torch.round(a)
    b=torch.sum(b**2)
    b.backward()
    print(a,a.grad)