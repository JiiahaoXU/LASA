
import torch
from torch import nn
import copy
import numpy as np

from utils.mask_help import *


def topk(vector, args):
    '''
    return the mask for topk of vector
    '''

    k_dim = int(args.com_p * args.dim)
    
    mask = torch.zeros_like(vector)
    # flat_abs = abs(flat)
    _, indices = torch.topk(vector**2, k_dim)
    # generate a mask, set topk as 1, otherwise 0
    mask[indices] = 1
    # mask = {k: mask[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return mask, mask*vector


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        # if key.split('.')[-1] == 'num_batches_tracked':
        #     continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_net_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the net parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

'''
Code is modified from https://github.com/JianXu95/SignGuard/tree/main/aggregators.
Modified by Jiahao Xu @ UNR.
Thanks for their contribution!
'''

def dnc(local_updates, global_model, args):
    ###########################
    ########## local ##########
    ###########################
    # print(len(local_updated_models))
    flat_local_updates = []

    iters = 1

    for param in local_updates:
        flat_param = parameters_dict_to_vector_flt(param)
        flat_local_updates.append(flat_param)
    grads = torch.stack(flat_local_updates, dim=0)
    
    num_users = args.num_selected_users
    num_byzs = args.num_attackers
    c = (num_users-1)/num_users
    all_set = set([i for i in range(num_users)])
    num_param = grads.shape[1]
    all_idxs = [i for i in range(num_param)]
    # grads[torch.isnan(grads)] = 0  # remove nan

    iters = 1
    num_spars = 1000
    benign_idx = all_set
    for it in range(iters):
        idx = np.random.choice(all_idxs, num_spars, replace=False)
        # set of gradients subsampled using indices in idx
        gradss = grads[:, idx]
        # Compute mean of input gradients
        mu = torch.mean(gradss, dim=0)
        # get centered input gradients
        gradss_c = gradss - mu
        # get the top right singular eigenvector
        try:
            U, S, V = torch.linalg.svd(gradss_c)
        except:
            V = None
        if V is not None:
            v = V[:, 0]
            # Compute outlier scores
            s = torch.mul((gradss - mu), v).sum(dim=1)**2
            dnc_idx = s.topk(int(num_users-c*num_byzs), dim=0, largest=False)[-1].cpu().numpy()
            benign_idx = benign_idx.intersection(set(dnc_idx))


    benign = list(benign_idx)

    gradssss = grads[benign].mean(dim=0)

    flat_local_updates = []
    # for param, i in zip(local_updates, range(args.num_selected_users)):
    # for idx in benign_idx1:
    #     param = local_updates[idx]
        # print(i)
    temp_model = copy.deepcopy(global_model)
    # temp_mask = generate_init_mask(global_model)

    # temp_mask = update_mask(param, temp_mask, args.sparsity)

    flat_param = gradssss

    # Clip
    # delta_norm = torch.norm(flat_param)
    # threshold = delta_norm / args.clip
    # if threshold > 1.0:
    #     flat_param = flat_param / threshold

    # Add DP noise
    if args.use_dp:
        delta_norm = torch.norm(flat_param)
        threshold = delta_norm / args.clip
        if threshold > 1.0:
            flat_param = flat_param / threshold

        args.sigma = args.noise_multiplier * args.clip / np.sqrt(args.num_selected_users)
        dp_noise = torch.normal(0, args.sigma, flat_param.shape).to(args.device)
        flat_param = flat_param + dp_noise

    # flat_local_param = parameters_dict_to_vector_flt(local_updated_models[i])
    flat_local_param = parameters_dict_to_vector_flt(temp_model)
    updated_local_para = flat_local_param + flat_param
    temp_model = vector_to_net_dict(updated_local_para, temp_model)

    # temp_mask = update_mask(temp_model, temp_mask, 0.8)

    flat_local_updates.append(temp_model)


    print("sigma ", args.sigma)
    ###########################
    ########## global #########
    ###########################

    global_model = copy.deepcopy(flat_local_updates[0])

    for i in range(1, len(flat_local_updates)):
        for key in global_model.keys():
            if key in flat_local_updates[i]:
                global_model[key].data = global_model[key].data + flat_local_updates[i][key].data
        # print(i)

    for key in global_model.keys():
        if key in flat_local_updates[0]:
            global_model[key].data = global_model[key].data / len(flat_local_updates)



    return global_model
    # return global_model, momentum, momentum2order, global_mask
