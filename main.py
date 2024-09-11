import torch, random, argparse, os, copy
import numpy as np
from algorithms.engine.fedavg_all import fedavg_all
from mmengine.config import Config


def merge_config(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="seed")
    parser.add_argument('--repeat', type=int, default=3, help='repeat index')
    parser.add_argument('--freeze_datasplit', type=int, default=0, help='freeze to save dict_users.pik or not')
    parser.add_argument('--sparsity', type=float, default=0.3, help='pre-defined sparsity')
    parser.add_argument('--num_attackers', type=int, default=20, help='Bayzatine attckers')
    parser.add_argument('--beta', type=float, default=0, help='ema')
    parser.add_argument('--attack', type=str, default='agrTailoredTrmean', help='attack method', choices=['agrTailoredTrmean', 'agrAgnosticMinMax', 'agrAgnosticMinSum', 'signflip_attack', 'noise_attack', \
                'random_attack', 'lie_attack', 'byzmean_attack', 'non_attack'])
    parser.add_argument('--defend', type=str, default='lasa', help='defend method', choices=['fedavg', 'signguard', 'dnc', 'lasa', 'bulyan', 'tr_mean', 'multi_krum', 'sparsefed', 'geomed'])

    parser.add_argument('--dataset', type=str, default='cifar', help='dataset')
    parser.add_argument('--lambda_n', type=float, default=1.0, help='reserver sparsity')
    parser.add_argument('--lambda_s', type=float, default=1.0, help='reserver sparsity')

    meta_args = parser.parse_args()

    if meta_args.dataset == 'sha':
        meta_args.dataset = 'shakespeare'
    
    if meta_args.dataset != 'noniidcifar' and meta_args.dataset != 'noniidcifar100':
        meta_args.alpha = -1

    meta_args.config_name = 'attack/%s/basee.yaml' % meta_args.dataset

    config_path = os.path.join('config/', meta_args.config_name)
    config = Config.fromfile(config_path)
    meta_args = merge_config(config, meta_args)

    meta_args.results_dir = './exp_results/%s/Attack_%s_Raito_%d/Defense_%s/' % (meta_args.dataset, str(meta_args.attack), meta_args.num_attackers, str(meta_args.defend))

    meta_args.num_attackers = int(meta_args.num_attackers * meta_args.num_selected_users / 100)

    if meta_args.defend == 'sparsefed':
        meta_args.com_p = 1 - meta_args.sparsity

    meta_args.device = torch.device('cuda')

    if not os.path.exists(meta_args.results_dir):
        os.makedirs(meta_args.results_dir)
    
    meta_args.exp_record = '%s/results.txt' % (meta_args.results_dir)

    # for reproducibility
    score_box = []
    poisoned_ratio_box = []
    for r in range(meta_args.repeat):
        args = copy.deepcopy(meta_args)
        print('############ Case '+ str(r) + ' ############')
        random.seed(args.seed+r)
        torch.manual_seed(args.seed+r)
        # torch.cuda.manual_seed(args.seed+args.repeat) # avoid
        np.random.seed(args.seed+r)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        best_result, poisoned_ratio = fedavg_all(args)
        score_box.append(best_result)
        poisoned_ratio_box.append(poisoned_ratio)
    print('repeated scores: ' + str(score_box))
    avg_score = np.average(score_box)
    print('avg of the scores ' + str(avg_score))
    
    print('repeated poisoned ratio: ' + str(poisoned_ratio_box))
    avg_poisoned_ratio = np.average(poisoned_ratio_box)
    print('avg of the poisoned ratios ' + str(avg_poisoned_ratio))

    if 'maskfeddp' in str(meta_args.config_name):
        sparsity = str(meta_args.sparsity)

    # String to write
    my_string = '---' * 10 + '\n' +\
                'dataset is: ' + str(meta_args.dataset) + ', ' + '\n' +\
                'attack is: ' + str(meta_args.attack) + ', ' + '\n' +\
                'defend is: ' + str(meta_args.defend) + ', ' + '\n' +\
                'DP: ' + str(args.use_dp) + ', ' + '\n' +\
                'num_attackers is: ' + str(meta_args.num_attackers) + ', ' + '\n' +\
                'sparsity is: ' + str(meta_args.sparsity) + ', ' + '\n' +\
                'repeated scores: ' + str(score_box) + ', ' + '\n' +\
                'avg of the scores ' + str(avg_score) + ', ' + '\n' +\
                'repeated poisoned ratios ' + str(poisoned_ratio_box) + ', ' + '\n' +\
                'avg of the poisoned ratios ' + str(avg_poisoned_ratio) + ', ' + '\n' +\
                '---'* 10 + '\n'

    # 'config name is: ' + str(meta_args.config_name) + ', ' + '\n' +\

    # Open the file in write mode
    with open(meta_args.exp_record, 'a') as f:
        # Write the string to the file
        f.write(my_string)