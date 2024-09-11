import torch


def generate_init_mask(global_model):
    mask = {}
    for key in global_model.keys():
        
        if len(global_model[key].size()) == 4 or len(global_model[key].size()) == 2:
            # Need to change the dtype, but now only for testing
            mask[key] = torch.ones_like(global_model[key], dtype=torch.float32, requires_grad=False).cuda()
    
    return mask


def generate_random_mask(global_model, sparsity):

    random_weights = {}
    random_mask = generate_init_mask(global_model)

    for key in random_mask.keys():
        random_weights[key] = torch.randn_like(random_mask[key])

    random_mask = update_mask(random_weights, random_mask, sparsity)
    
    return random_mask


def update_mask(model, mask, sparsity):
    
    if sparsity == 0.0:
        for key in model.keys():
            if key not in mask: 
                continue
            else:
                mask[key] = torch.ones_like(mask[key]).float() 

        return mask

    weight_abs = []
    for key in model.keys():
        if key not in mask: 
            continue
        else:    
            weight_abs.append(torch.abs(model[key]))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
    num_params_to_keep = int(len(all_scores) * (1 - sparsity))

    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    for key in model.keys():
        if key not in mask: 
            continue
        else:
            mask[key] = ((torch.abs(model[key])) > acceptable_score).float()  # must be > to prevent acceptable_score is zero, leading to dense tensors

    return mask
    

def apply_mask(model, mask):

    for key in model.keys():
        if key in mask:
            model[key].data = model[key].data * mask[key]

    return model


