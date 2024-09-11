# coding: utf-8
'''
Code is modified from https://github.com/JianXu95/SignGuard/tree/main/aggregators.
Modified by Jiahao Xu @ UNR.
Thanks for their contribution!
'''


import math
import torch
import numpy as np

# ---------------------------------------------------------------------------- #
# naive gradient attacks

def nan_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    device = byz_grads[0][0].device
    # Generate the non-finite Byzantine gradient
    nan_grad = torch.empty_like(byz_grads[0]).to(device)
    nan_grad.copy_(torch.tensor((math.nan,), dtype=nan_grad.dtype))
    return [nan_grad] * num_byzs


def zero_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    device = byz_grads[0][0].device
    # Return this Byzantine gradient 'num_byzs' times
    return [torch.zeros_like(byz_grads[0]).to(device)] * num_byzs


def random_attack(all_updates, args, malicious_attackers_this_round):
    

    # selected_idxs_1 = list(np.random.choice(range(args.num_selected_users), args.num_attackers, replace=False))

    for i in range(malicious_attackers_this_round):
        for key in all_updates[i].keys():
            all_updates[i][key] = 0.5 * torch.randn(all_updates[i][key].size()).to(all_updates[i][key].device)

    #     for key in all_updates[i].keys():
    #         all_updates[i][key] = 0.5 * torch.randn(all_updates[i][key].size()).to(all_updates[i][key].device)



    return all_updates


def noise_attack(all_updates, args, malicious_attackers_this_round):

    for i in range(malicious_attackers_this_round):

        for key in all_updates[i].keys():
            if 'num_batches_tracked' in key:
                continue
            all_updates[i][key] += 0.5 * torch.randn(all_updates[i][key].size()).to(all_updates[i][key].device)
            # all_updates[idx][key] += torch.randn(all_updates[idx][key].size()).to(all_updates[idx][key].device)

    return all_updates


def signflip_attack(all_updates, args, malicious_attackers_this_round):

    # selected_idxs_1 = list(np.random.choice(range(args.num_selected_users), args.num_attackers, replace=False))
    
    for i in range(malicious_attackers_this_round):

        for key in all_updates[i].keys():
            all_updates[i][key] *= -1

    return all_updates

    # num_byzs = len(byz_grads)
    # if num_byzs == 0:
    #     return list()
    # return [-1.0*x for x in byz_grads]


def non_attack(all_updates, *args, **kwargs):

    return all_updates