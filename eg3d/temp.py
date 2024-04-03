import torch
import dnnlib
import legacy

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all), f'Missing source tensor: {name}'
        if name in src_tensors:
            tensor.requires_grad_(False)
            print('name: ', name)
            print('src size: ', src_tensors[name].size())
            print('dest size: ', tensor.size())
            tensor.copy_(src_tensors[name].detach()).requires_grad_(src_tensors[name].requires_grad)

def list_to_dict(myList):
    ret = {}
    for (a, b) in myList:
        ret[a] = b
    return ret

afhq = '/playpen-nas-ssd/awang/eg3d/eg3d/models/afhqcats512-128.pkl'
ffhq = '/playpen-nas-ssd/awang/eg3d/eg3d/models/ffhqrebalanced512-128.pkl'

with dnnlib.util.open_url(ffhq) as f:
    ffhq = legacy.load_network_pkl(f)
with dnnlib.util.open_url(afhq) as f:
    afhq = legacy.load_network_pkl(f)

for name in ['G', 'D','G_ema']:
    ffhq_dict = list_to_dict(named_params_and_buffers(ffhq[name]))
    afhq_dict = list_to_dict(named_params_and_buffers(afhq[name]))
    print('module name: ', name)
    for param_name in ffhq_dict:
        if 'superresolution' in param_name:
            if ffhq_dict[param_name].size() != afhq_dict[param_name].size():
                print('name: ', param_name)
                print('size of ffhq: ', ffhq_dict[param_name].size())
                print('size of afhq: ', afhq_dict[param_name].size())
