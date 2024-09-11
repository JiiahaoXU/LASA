'''
Code is modified from https://github.com/JianXu95/SignGuard/tree/main.
Modified by Jiahao Xu @ UNR.
Thanks for their contribution!
'''


import torch


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


def lie_attack(all_updates, args, malicious_attackers_this_round, z=0.5):
    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:malicious_attackers_this_round]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        # all_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)
        all_attack_updates_flatten.append(update)

    all_updates_flatten = torch.stack(all_attack_updates_flatten)
    avg = torch.mean(all_updates_flatten, dim=0)
    std = torch.std(all_updates_flatten, dim=0)
    mal_update = avg - z * std

    mal_dict = vector_to_net_dict(mal_update, all_updates[0])

    if malicious_attackers_this_round > 0:
        for i in range(malicious_attackers_this_round):
            all_updates[i] = mal_dict

    return all_updates


def byzmean_attack(all_updates, args, malicious_attackers_this_round, z=0.5):
    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten.append(update)

    all_updates_flatten = torch.stack(all_attack_updates_flatten)
    avg = torch.mean(all_updates_flatten, dim=0)
    std = torch.std(all_updates_flatten, dim=0)
    mal_update = avg - z * std

    # args.num_attackers
    m1 = int(0.5 * malicious_attackers_this_round)
    if malicious_attackers_this_round > 1:

        m2 = malicious_attackers_this_round - m1

    else:
        m2 = None

    # print(m1, m2)
    mal_dict_1 = vector_to_net_dict(mal_update, all_updates[0])

    selected_idxs_1 = [x for x in range(malicious_attackers_this_round)]
    # print(selected_idxs_1)
    for idx in selected_idxs_1[:m1]:
        all_updates[idx] = mal_dict_1

    if m2 is not None:
        byz_grad2 = ((args.num_selected_users - malicious_attackers_this_round -m1)*mal_update-torch.sum(all_updates_flatten, dim=0))/m2
        mal_dict_2 = vector_to_net_dict(byz_grad2, all_updates[0])
        for idx in selected_idxs_1[m1:]:
            all_updates[idx] = mal_dict_2

    return all_updates


