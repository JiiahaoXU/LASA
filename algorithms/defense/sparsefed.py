'''
Implementation of SparseFed: Mitigating Model Poisoning Attacks in Federated Learning with Sparsification.
'''

import torch
import torch
import copy



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
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def no_defence_balance_(ave_model_update, global_model, local_updates):
    '''
    ave_model_update is a vector; others are dicts; 
    change the corresponding parts in fang and flame
    '''
    pointer = 0
    model_update = copy.deepcopy(global_model)

    for key, param in model_update.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        num_param = param.numel()
        param.data = ave_model_update[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

    for var in global_model:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_model[var] = local_updates[0][var]
            continue
        global_model[var] += model_update[var]
    return global_model

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

def sparsefed(local_updates, global_model, args, momentum=None, error=None):
    momentum = None
    # flatten models
    local_updates_vector = []
    for param in local_updates:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_updates_vector = parameters_dict_to_vector_flt(param)[None, :] if not len(local_updates_vector) else torch.cat((local_updates_vector, parameters_dict_to_vector_flt(param)[None, :]), 0)
    # aggregate local model updates
    ave_local_update = torch.mean(local_updates_vector, 0)
    
    # momentum update on the server side # remember to add this to other algorithms
    if momentum is None:
        momentum = torch.zeros_like(ave_local_update)
        error = torch.zeros_like(ave_local_update)
    momentum = args.global_momentum * momentum + ave_local_update
    # error feedback
    compensated_local_update = error + momentum
    # compute topk mask
    mask, sparsed_local_update = topk(compensated_local_update, args)
    
    # update error
    error = compensated_local_update - sparsed_local_update
    # convert vector to net and update global model
    global_model = no_defence_balance_(sparsed_local_update, global_model, local_updates)
    
    return global_model, momentum, error
