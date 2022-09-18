#%% 

import os, subprocess
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
            for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

import torch
import torch.nn as nn
import argparse
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


#%%
#**********************Step1: Define available device****************************

def get_device(device_name):
    # device_name = "CUDA"
    assert device_name in ["CUDA", "Multi-thread CPU", "Single-thread CPU"]
    if device_name == 'CUDA' and torch.cuda.is_available():
        device = torch.device("cuda")
        print('os.environ["CUDA_VISIBLE_DEVICES"]', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        device = torch.device("cpu")

        if device_name == "Single-thread CPU":
            os.environ["OMP_NUM_THREADS"] = '1'
            # Use command OMP_NUM_THREADS=1 python operator_profile.py
        elif device_name == "Multi-thread CPU":
            os.environ["OMP_NUM_THREADS"] = '32'
        print('os.environ["OMP_NUM_THREADS"] = '+os.environ["OMP_NUM_THREADS"])
    return device     


#%%
#**********************Step2: Define profiler****************************
from profiler import MyTimeit, MyFlops

            
            
#%%
#**********************Step3: Define Main****************************


def main(test_name, inp, device, opt):
    
    if isinstance(inp, torch.Tensor):
        inp = inp.to(device)
    if isinstance(test_name, nn.Module):
        test_name   = test_name.to(device)
    if isinstance(inp, torch.Tensor):
        print('size of tensor '+str(inp.shape))
    print('use device ', device)

    with torch.no_grad(): out = test_name(lr=inp)
    if isinstance(out, torch.Tensor):
        print('output shape is', out.shape)
    elif isinstance(out, list) or isinstance(out, tuple):
        print('output shape is', out[0].shape)
    del out
    torch.cuda.empty_cache()
    if isinstance(test_name, nn.Module):
        test_name = test_name.eval()
    # print('size of tensor'+str(inp.shape))
    # print('use device', device)

    with torch.no_grad():
        
        # for profiler in ( MyTimeit('time'),
            # MyTimeit('torchprofile19', path='./torchprofiler/'+opt['name']),):
        for profiler in (
                        MyFlops(mode='ptflops'),
                        MyTimeit('time'),
                        ):
        # for profiler in (MyFlops(mode='thop'),):
                        # thop ptflops
        # for profiler in (MyTimeit('line'),):
            # for function in test_list:
                print('\n test function name: '+str(test_name.__class__))
                # check if it is a module
                new_function = profiler(test_name)
                out = new_function(inp)
                del out
                torch.cuda.empty_cache()

    print("*****************************************************************************")


# %%

if __name__ == '__main__':
    # import options.options as option
    from basicsr.utils.options import *
    from basicsr.models import build_model
    import options.options as option

    print("__file__", __file__)
    import os.path as osp
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    # opt, args = parse_options(root_path, is_train=False)
    # import Experimental_root.archs
    # import Experimental_root.models
    # test_path = "/home/chenyangqi/disk1/Video-Enhancement-Playground/options/train/jpg_inter/1215_train_jpg_inter_dev.yml"
    opt = "options/test/0222_test_Rescaling_DF2K_4X_HCFlow_jpeg.yml"
    # test_path = "/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/options/train/jpg_inter/1229_RescaleNet_V1_arch_pos_encoding_lr02_323.yml"
    
    opt = option.parse(opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    
    from models import create_model
    model = create_model(opt)


# %%
    # def upscale(self, LR_img, scale, gaussian_scale=1):
    # inp = torch.randn(1, 3, 3840//64*64, 2160//64*64)
    w = 5760//64*64//4
    h = 3240//64*64//4
    device_name = "CUDA"
    print("device_name", device_name)
    device      = get_device(device_name)
    print("device", device)
#%%
    decoder_function = model.netG
    print(decoder_function)
    
    # encoder_function = model
    inp = torch.randn(1, 3, h, w).cuda() # with posi encoding
    with torch.cuda.amp.autocast(True):
        main(decoder_function, inp, device, opt)

    
# %%
    inp = torch.randn(1, 64, h//8, w//8)
    print(model.net_g.ydecoder)
    decoder_function = model.net_g.ydecoder
    # encoder_function = model
    
    with torch.cuda.amp.autocast(True):
        main(decoder_function, inp, device, opt)
# %%
    inp = torch.randn(1, 128, h//16, w//16)
    print(model.net_g.brdecoder)
    decoder_function = model.net_g.brdecoder
    # encoder_function = model
    
    with torch.cuda.amp.autocast(True):
        main(decoder_function, inp, device, opt)
# %%
