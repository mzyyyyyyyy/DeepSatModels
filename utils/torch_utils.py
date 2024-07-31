import torch
import os
import glob
import sys


def load_from_checkpoint(net, checkpoint, partial_restore=False, modified_model=False, device=None):
    
    assert checkpoint is not None, "no path provided for checkpoint, value is None"
    if os.path.isdir(checkpoint):
        checkpoint = max(glob.iglob(checkpoint + '/*.pth'), key=os.path.getctime)
        print("loading model from %s" % checkpoint)
        if device is None:
            saved_net = torch.load(checkpoint)
        else:
            saved_net = torch.load(checkpoint, map_location=device)
        saved_net = torch.load(checkpoint, map_location=device)
    elif os.path.isfile(checkpoint):
        print("loading model from %s" % checkpoint)
        if device is None:
            saved_net = torch.load(checkpoint)
        else:
            saved_net = torch.load(checkpoint, map_location=device)
    else:
        raise FileNotFoundError("provided checkpoint not found, does not mach any directory or file")
    
    if partial_restore:
        net_dict = net.state_dict()
        saved_net = {k: v for k, v in saved_net.items() if (k in net_dict) and (k not in ["linear_out.weight", "linear_out.bias"])}
        print("params to keep from checkpoint:")
        print(saved_net.keys())
        extra_params = {k: v for k, v in net_dict.items() if k not in saved_net}
        print("params to randomly init:")
        print(extra_params.keys())
        for param in extra_params:
            saved_net[param] = net_dict[param]

    if modified_model:
        # 修改 temporal_token 并重新命名为与 TSViT_single_token 类中定义的一致
        temporal_token = saved_net.pop('temporal_token')
        temporal_token_single = temporal_token.mean(dim=1, keepdim=True)
        saved_net['temporal_token_single'] = temporal_token_single
        
        # 修改 mlp_head 并重新命名为与 TSViT_single_token 类中定义的一致
        
        # print(saved_net['mlp_head.0.weight'].shape)
        # print(saved_net['mlp_head.0.bias'].shape)
        # print(saved_net['mlp_head.1.weight'].shape)
        # print(saved_net['mlp_head.1.bias'].shape)

        # print(net.state_dict()['mlp_head_single.0.weight'].shape)
        # print(net.state_dict()['mlp_head_single.0.bias'].shape)
        # print(net.state_dict()['mlp_head_single.1.weight'].shape)
        # print(net.state_dict()['mlp_head_single.1.bias'].shape)

        mlp_0_w = saved_net.pop('mlp_head.0.weight')
        saved_net['mlp_head_single.0.weight'] = mlp_0_w

        mlp_0_b = saved_net.pop('mlp_head.0.bias')
        saved_net['mlp_head_single.0.bias'] = mlp_0_b

        mlp_1_w = saved_net.pop('mlp_head.1.weight')
        mlp_1_w_single = mlp_1_w.repeat_interleave(19, dim=0)[:76]
        saved_net['mlp_head_single.1.weight'] = mlp_1_w_single

        mlp_1_b = saved_net.pop('mlp_head.1.bias')
        mlp_1_b_single = mlp_1_b.repeat_interleave(19, dim=0)[:76]
        saved_net['mlp_head_single.1.bias'] = mlp_1_b_single                


    
    net.load_state_dict(saved_net, strict=True)
    return checkpoint


def get_net_trainable_params(net):
    try:
        trainable_params = net.trainable_params
    except AttributeError:
        trainable_params = list(net.parameters())
    print("Trainable params shapes are:")
    print([trp.shape for trp in trainable_params])
    return trainable_params
    
    
def get_device(device_ids, allow_cpu=True):
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % device_ids[0])
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        sys.exit("No allowed device is found")
    return device

