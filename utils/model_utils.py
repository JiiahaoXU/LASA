from model.cnn import CNNFmnist
from model.recurrent import RNN_FedShakespeare
from model.resnet import ResNet18
import torch
import copy

################################### model setup ########################################
def model_setup(args):

    if args.dataset in ['fmnist', 'mnist', 'femnist']:
        args.model = 'cnnfmnist'
    elif args.dataset in ['cifar', 'noniidcifar', 'cifar100', 'noniidcifar100']:
        args.model = 'resnet18'
    elif args.dataset == 'shakespeare':
        args.model = 'rnnshakespeare'

    if args.model == 'cnnfmnist':
        net_glob = CNNFmnist(args=args).to(args.device)

    elif args.model == 'resnet18' and 'cifar100' not in args.dataset:
        net_glob = ResNet18().to(args.device)
    elif args.model == 'resnet18' and 'cifar100' in args.dataset:
        net_glob = ResNet18(num_classes=100).to(args.device)

    elif args.model == 'rnnshakespeare':
        net_glob = RNN_FedShakespeare().to(args.device)
    else:
        exit('Error: unrecognized model')
    global_model = copy.deepcopy(net_glob.state_dict())
    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s


