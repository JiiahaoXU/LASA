

from __future__ import print_function
import numpy as np
import torch
import torch.nn.parallel
from geom_median.torch import compute_geometric_median 
import copy

def multi_krum(all_updates, n_attackers=10, multi_k=False):
    num_users = len(all_updates)
    # flatten model parameters
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates_flatten
    all_indices = np.arange(num_users)

    
    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        distances = torch.sort(distances, dim=1)[0]
        
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
    
        if not multi_k:
            break
    aggregate = torch.mean(candidates, dim=0) # mean for multi-krum, this line doesn't influence krum
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model, np.array(candidate_indices)

def bulyan(all_updates, n_attackers=5):
    num_users = len(all_updates)
    # flatten model parameters
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates_flatten
    all_indices = np.arange(num_users)

    while len(bulyan_cluster) < (num_users - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        # print(distances)
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]
    
    # aggregate = torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)
    aggregate = torch.mean(sorted_params[n_attackers:-n_attackers], 0) # trimmed mean

    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model, np.array(candidate_indices)

def tr_mean(all_updates, n_attackers=10):
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    sorted_updates = torch.sort(all_updates_flatten, 0)[0]
    aggregate = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model

def median(all_updates, n_attackers=10):
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    sorted_updates = torch.sort(all_updates_flatten, 0)[0]
    aggregate = torch.median(sorted_updates[n_attackers:-n_attackers], 0)[0] if n_attackers else torch.mean(sorted_updates,0)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model

#coordinate_median
def coordinate_median(all_updates):
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    sorted_updates = torch.sort(all_updates_flatten, 0)[0]
    aggregate = torch.median(sorted_updates, 0)[0]
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model

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



def geomed(local_updates, global_model, args):
    ###########################
    ########## local ##########
    ###########################
    # print(len(local_updated_models))
    all_masks = []
    flat_local_updates = []

    # device = local_updates[0][0].device
    # num_users = args.num_selected_users
    all_set = set([i for i in range(args.num_selected_users)])
    iters = 1

    for param in local_updates:
        flat_param = parameters_dict_to_vector_flt(param)
        flat_local_updates.append(flat_param)
    grads = torch.stack(flat_local_updates, dim=0)
    
    weights = torch.ones(len(grads)).cuda()
    gw = compute_geometric_median(flat_local_updates, weights).median
    iter = 2 if 'cifar' not in args.dataset else 0
    for i in range(iter):
        weights = torch.mul(weights, torch.exp(-1.0*torch.norm(grads-gw, dim=1)))
        gw = compute_geometric_median(flat_local_updates, weights).median
    
    # global_grad = grads_clip[benign_idx].mean(dim=0)
    # global_grad = grads_clip[benign_idx].mean(dim=0)


    flat_local_updates = []
    # for param, i in zip(local_updates, range(args.num_selected_users)):
    # for idx in benign_idx1:
    #     param = local_updates[idx]
        # print(i)
    temp_model = copy.deepcopy(global_model)
    # temp_mask = generate_init_mask(global_model)

    # temp_mask = update_mask(param, temp_mask, args.sparsity)

    flat_param = gw

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
    # print(flat_local_param.shape, flat_param.shape)
    updated_local_para = flat_local_param + flat_param
    temp_model = vector_to_net_dict(updated_local_para, temp_model)

    # temp_mask = update_mask(temp_model, temp_mask, 0.8)
    # print(type(temp_model))
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
