'''
Efficient and Private Federated Learning with Sparsely DP in Federated Learning
'''
import torch
from torch import nn
import copy
import numpy as np

from utils.mask_help import *

from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth


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

def signguard(local_updates, global_model, args):
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
    grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
    if np.any(np.isnan(grad_l2norm)):
        grad_l2norm = np.where(np.isnan(grad_l2norm), 0, grad_l2norm)
    norm_max = grad_l2norm.max()
    norm_med = np.median(grad_l2norm)
    benign_idx1 = all_set
    benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm > 0.1*norm_med)]))
    benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < 3.0*norm_med)]))
    # print(grad_l2norm)
    ## sign-gradient based clustering
    num_param = grads.shape[1]
    num_spars = int(0.1 * num_param)
    benign_idx2 = all_set

    dbscan = 0
    # meanshif = int(1-dbscan)

    for it in range(iters):
        idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
        gradss = grads[:, idx:(idx+num_spars)]
        sign_grads = torch.sign(gradss)
        sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
        sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
        sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
        pos_max = sign_pos.max()
        pos_feat = sign_pos / (pos_max + 1e-8)
        zero_max = sign_zero.max()
        zero_feat = sign_zero / (zero_max + 1e-8)
        neg_max = sign_neg.max()
        neg_feat = sign_neg / (neg_max + 1e-8)

        feat = [pos_feat, zero_feat, neg_feat]
        sign_feat = torch.stack(feat, dim=1).cpu().numpy()

        # 
        if dbscan:
            clf_sign = DBSCAN(eps=0.05, min_samples=2).fit(sign_feat)
            labels = clf_sign.labels_
            n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
            num_class = []
            for i in range(n_cluster):
                num_class.append(np.sum(labels==i))
            benign_class = np.argmax(num_class)
            benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
        else:
            bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=args.num_selected_users)
            # print(bandwidth)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
            ms.fit(sign_feat)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            labels_unique = np.unique(labels)
            n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
            num_class = []
            for i in range(n_cluster):
                num_class.append(np.sum(labels==i))
            benign_class = np.argmax(num_class)
            benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
    print('Norm malicious:', all_set - benign_idx1)
    print('Sign malicious:', all_set - benign_idx2)
    benign_idx = list(benign_idx2.intersection(benign_idx1))
    print(benign_idx)
    # print('----')
    grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
    norm_clip = grad_norm.median(dim=0)[0].item()
    grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
    grads_clip = (grads/grad_norm)*grad_norm_clipped
    
    global_grad = grads[benign_idx].mean(dim=0)
    # global_grad = grads_clip[benign_idx].mean(dim=0)


    flat_local_updates = []
    temp_model = copy.deepcopy(global_model)

    flat_param = global_grad

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