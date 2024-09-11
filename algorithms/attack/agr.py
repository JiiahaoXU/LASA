import torch
import numpy as np

# modified code from ndss 21
'''
MIN-MAX attack
'''
def agrAgnosticMinMax(all_updates, args, malicious_attackers_this_round, model_re=None, dev_type='std', threshold=10):
    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:malicious_attackers_this_round]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_attack_updates_flatten, 0)

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_attack_updates_flatten, 0)

    lamda = torch.Tensor([threshold]).float().to(args.device)
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_attack_updates_flatten:
        distance = torch.norm((all_attack_updates_flatten - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_attack_updates_flatten - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        
        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
    
    # use the same poisoned updates
    mal_update = mal_update.unsqueeze(0).repeat(malicious_attackers_this_round, 1)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    # mal_models=[]
    for i in range(malicious_attackers_this_round):
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}

    return all_updates

'''
MIN-SUM attack
'''

def agrAgnosticMinSum(all_updates, args, malicious_attackers_this_round, model_re=None, dev_type='std', threshold=10):

    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:malicious_attackers_this_round]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_attack_updates_flatten, 0)

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_attack_updates_flatten, 0)
    
    lamda = torch.Tensor([threshold]).float().to(args.device)
    # print(lamda)
    threshold_diff = 1e-5
    # threshold_diff = 0.02
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_attack_updates_flatten:
        distance = torch.norm((all_attack_updates_flatten - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    scores = torch.sum(distances, dim=1)

    # min_score = torch.min(scores)
    min_score = torch.max(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_attack_updates_flatten - mal_update), dim=1) ** 2
        score = torch.sum(distance)
        
        if score <= min_score:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print(lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)
    
    # use the same poisoned updates
    mal_update = mal_update.unsqueeze(0).repeat(malicious_attackers_this_round, 1)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    # mal_models=[]
    for i in range(malicious_attackers_this_round):
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}

    return all_updates


def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    return out

def multi_krum(updates, n_attackers, multi_k):
    '''
    multi_k = False for fang's attack
    '''
    num_users = len(updates)
    
    candidates = []
    candidate_indices = []
    remaining_updates = updates
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
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)

def bulyan(all_updates, n_attackers):
    nusers = all_updates.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
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

    return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)

def agrTailoredTrmean(all_updates, args, malicious_attackers_this_round, dev_type='std', threshold=10):
    # n_attackers = args.num_attackers
    n_attackers = max(1, args.num_attackers**2//args.num_selected_users)
    
    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:malicious_attackers_this_round]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_attack_updates_flatten, 0)
    
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_attack_updates_flatten, 0)

    lamda = torch.Tensor([threshold]).float().to(args.device) #compute_lambda_our(all_updates, model_re, n_attackers)
    # print(lamda)
    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0 
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_attack_updates_flatten), 0)

        agg_grads = tr_mean(mal_updates, n_attackers)
        
        loss = torch.norm(agg_grads - model_re)
        
        if prev_loss < loss:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
        
    mal_update = (model_re - lamda_succ * deviation)

    # use the same poisoned updates
    mal_update = mal_update.unsqueeze(0).repeat(malicious_attackers_this_round, 1)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    # mal_models=[]
    for i in range(malicious_attackers_this_round):
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}
    
    return all_updates

def agrTailoredMedian(all_updates, args, dev_type='std', threshold=10):
    # n_attackers = args.num_attackers
    n_attackers = max(1, args.num_attackers**2//args.num_selected_users)

    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:args.num_attackers]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_attack_updates_flatten, 0)
    
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_attack_updates_flatten, 0)

    lamda = torch.Tensor([threshold]).float().to(args.device) #compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0 
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_attack_updates_flatten), 0)

        agg_grads = torch.median(mal_updates, 0)[0]
        
        loss = torch.norm(agg_grads - model_re)
        
        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
        
    mal_update = (model_re - lamda_succ * deviation)

    # use the same poisoned updates
    mal_update = mal_update.unsqueeze(0).repeat(args.num_attackers, 1)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    # mal_models=[]
    for i in range(args.num_attackers):
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}
    
    return all_updates

def agrTailoredKrumBulyan(all_updates, args, dev_type='std'):
    # n_attackers = args.num_attackers
    n_attackers = max(1, args.num_attackers**2//args.num_selected_users)
    
    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:args.num_attackers]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_attack_updates_flatten, 0)
    
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_attack_updates_flatten, 0)

    lamda = torch.Tensor([3.0]).cuda().float().to(args.device)

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_attack_updates_flatten), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    # use the same poisoned updates
    mal_update = mal_update.unsqueeze(0).repeat(args.num_attackers, 1)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    # mal_models=[]
    for i in range(args.num_attackers):
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}
    
    return all_updates