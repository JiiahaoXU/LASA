import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from utils.mask_help import *


def test_img(net_g, datatest, args):
    net_g = copy.deepcopy(net_g).to(args.device)
    loss_func = nn.CrossEntropyLoss()
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_batch_size)

    for index, (data, target) in enumerate(data_loader):

                
        data, target = data.to(args.device), target.to(args.device)
        # print(data[0], target[0])
        
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += loss_func(log_probs, target).item()
        # get the index of the max log-probability
        # print(log_probs[0])
        # print(torch.max(log_probs[0], -2))
        # exit()
        if args.dataset == 'shakespeare':
            _, predicted = torch.max(log_probs, -2)
            # correct += predicted[:,-1].eq(target[:,-1]).sum()
            correct += predicted.eq(target).sum()/target.shape[1]
            # print(predicted[:,-1].eq(target[:,-1]))
        else:
            _, predicted = torch.max(log_probs, -1)
            correct += predicted.eq(target).sum()

    test_loss /= len(datatest)
    # print(correct, len(datatest))
    accuracy = 100.00 * correct.item() / len(datatest)
    # exit()
    return accuracy, test_loss


