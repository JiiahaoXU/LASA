
import torch
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


def lasa(local_updates, global_model, args):
    ###########################
    ########## local ##########
    ###########################

    local_updates_ = []
    for i in range(len(local_updates)):
        vector = parameters_dict_to_vector_flt(local_updates[i])
        if vector.isnan().any():
            continue
        local_updates_.append(local_updates[i])

    local_updates = local_updates_

    flat_local_updates = []

    for i in range(len(local_updates)):
        vector = parameters_dict_to_vector_flt(local_updates[i])
        if vector.isnan().any():
            continue
        flat_local_updates.append(vector)

    flat_all_grads = torch.stack(flat_local_updates, dim=0)
    grad_norm = torch.norm(flat_all_grads, dim=1).reshape((-1, 1))
    norm_clip = grad_norm.median(dim=0)[0].item()
    grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
    grads_clip = (flat_all_grads/grad_norm)*grad_norm_clipped

    del grad_norm, norm_clip, grad_norm_clipped

    clipped_local_updates = []

    for i in range(len(local_updates)):
        net = vector_to_net_dict(grads_clip[i], local_updates[i])
        clipped_local_updates.append(net)

    # Pre-aggregation sparsification
    for i in range(len(local_updates)):
        global_mask = generate_init_mask(local_updates[i])
        global_mask = update_mask(local_updates[i], global_mask, args.sparsity)
        local_updates[i] = apply_mask(local_updates[i], global_mask)


    key_mean_weight = {}
    for key in local_updates[0].keys():
        if 'num_batches_tracked' in key:
            continue
        key_flat_para = []
        all_set = set([i for i in range(args.num_selected_users)])
        for param in local_updates:
            flat_param = param[key].flatten()
            # print(flat_param.numel())
            key_flat_para.append(flat_param)
        grads = torch.stack(key_flat_para, dim=0)


        # Norm check
        grad_l2norm = torch.norm(grads.float(), dim=1).cpu().numpy()
        norm_med = np.median(grad_l2norm)
        norm_std = np.std(grad_l2norm)

        # Calcualte MZ-score
        for i in range(len(grad_l2norm)):
            grad_l2norm[i] = np.abs((grad_l2norm[i] - norm_med) / norm_std)

        benign_idx1 = all_set.copy()
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < args.lambda_n)]))


        ##################
        # Sign check
        layer_sign = []
        for i in range(len(local_updates)):
            layer_sign.append(0.5 * (1 + torch.sum(torch.sign(local_updates[i][key])) / torch.sum(torch.abs(torch.sign(local_updates[i][key]))) * (1 - args.sparsity)).item())

        benign_idx2 = all_set.copy()
        if len(layer_sign) > 0:
            median = np.median(layer_sign)
            std = np.std(layer_sign)

            # Calcualte MZ-score
            for i in range(len(layer_sign)):
                layer_sign[i] = np.abs((layer_sign[i] - median) / std)
            benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(torch.tensor(layer_sign).cpu().numpy() < args.lambda_s)]))


        benign_idx = list(benign_idx2.intersection(benign_idx1))
        if len(benign_idx) == 0:
            benign_idx = list(all_set)
        
        # Layer-wise adaptive aggregation
        key_mean_weight[key] = torch.mean(torch.stack([clipped_local_updates[i][key] for i in benign_idx], dim=0), dim=0)

    for key in key_mean_weight.keys():
        if 'num_batches_tracked' in key:
            continue
        global_model[key].data += key_mean_weight[key].data
    

    return global_model


