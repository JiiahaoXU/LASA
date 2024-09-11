import copy
import numpy as np
import time, math
import torch
from torch.utils.data import DataLoader

from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from utils.mask_help import *
from test import test_img

from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average

from ..defense.byzantine_robust_aggregation import multi_krum, bulyan, tr_mean, geomed
from ..defense.sparsefed import sparsefed

from ..defense.lasa import lasa
from ..defense.signguard import signguard
from ..defense.dnc import dnc

import time

from ..attack import attack

def fedavg_all(args):
    ################################### hyperparameter setup ########################################
    print("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60])
    # args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users = load_partition(args)
    args, dataset_train, dataset_test, dataset_val, _, dict_users = load_partition(args)
    print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test) + len(dataset_val)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of validation data:{}'.format(len(dataset_val)))
    # print('num. of public data:{}'.format(len(dataset_public)))
    print('num. of users:{}'.format(len(dict_users)))

    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users)) # max 525, min 3


    print('average num. of samples per user:{}'.format(sample_per_users))
    

    
    print("{:<50}".format("-" * 15 + " model setup " + "-" * 50)[0:60])
    args, net_glob, global_model, args.dim = model_setup(args)

    print('model dim:', args.dim)

    ###################################### model initialization ###########################
    t1 = time.time()
    train_loss, test_acc = [], []
    print("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60])
    # initialize data loader for training and/or public dataset
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i])
        ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_list.append(ldr_train)

    net_glob.train()

    best_test_accuracy = 0

    nsr = 0.0

    attack_flag = False
    defend_flag = False
    if hasattr(args, 'attack'):
        if args.attack != 'None':
            attack_flag = True
        else:
            args.attack = None
            args.num_attackers = 0
    else:
        args.attack = None
        args.num_attackers = 0
    
    if hasattr(args, 'defend'):
        if args.defend != 'None':
            defend_flag = True
        else:
            args.defend = None
    else:
        args.defend = None

    # sampling attackers' id
    if args.attack:
        attacked_idxs = list(np.random.choice(range(args.num_users), int(args.num_attackers/args.num_selected_users*args.num_users), replace=False))
    overall_attack_ratio = []

    if args.attack != 'dynamic':
        attack_method = attack(args.attack)

    for t in range(args.round):
        if args.attack:
            gt_attack_cnt = 0

        ## learning rate decaying
        if args.dataset == 'shakespeare' or args.dataset == 'femnist':
            if (t+1) % 10 == 0:
                args.local_lr = args.local_lr * args.decay_weight
        else:
            args.local_lr = args.local_lr * args.decay_weight

        if args.num_attackers == 99:
            upper = int(25 * args.num_selected_users / 100)
            args.num_attackers = np.random.randint(10, upper+1)
            attacked_idxs = list(np.random.choice(range(args.num_users), int(args.num_attackers/args.num_selected_users*args.num_users), replace=False))

            print('At this round, attack ratio is %s' % args.num_attackers)

        ############################################################# FedAvg ##########################################
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))

        local_models, local_losses, local_updates, malicious_updates, delta_norms= [], [], [], [], []
        
        if args.dataset == 'shakespeare':
            num_of_label = 89
        elif args.dataset == 'femnist':
            num_of_label = 61
        else:
            num_of_label = 9

        local_solver = LocalUpdate(args=args)

        for i in selected_idxs:
            start = time.time()

            ################## <<< Attack Point 1: train with poisoned data
            net_glob.load_state_dict(global_model)
            
            if attack_flag and i in attacked_idxs:
                gt_attack_cnt += 1
                local_model, local_loss = local_solver.local_sgd_mome(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i], attack_flag=attack_flag, attack_method=args.attack, num_of_label=num_of_label)
            else:
                local_model, local_loss = local_solver.local_sgd_mome(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
            
            

            local_losses.append(local_loss)
            # compute model update
            model_update = {k: local_model[k] - global_model[k] for k in global_model.keys()}


            # compute model update norm
            end = time.time()

            # clipping local model 
            if defend_flag:
                if args.defend in ['sparsefed', 'tr_mean', 'krum', 'bulyan', 'fedavg', 'geomed'] and 'cifar' not in args.dataset:
                    delta_norm = torch.norm(torch.cat([torch.flatten(model_update[k]) for k in model_update.keys()]))
                    delta_norms.append(delta_norm)
                    threshold = delta_norm / args.clip
                    if threshold > 1.0:
                        for k in model_update.keys():
                            model_update[k] = model_update[k] / threshold
            # collecting local models
            # 32 bits * args.dim, {(index, param)}: k*32+log2(d); 32->4; 
            if attack_flag and i in attacked_idxs:
                malicious_updates.append(model_update)
            else:
                local_updates.append(model_update)

            #
        # calculate_sparsity(local_model)
        # add malicious update to the start of local updates
        malicious_attackers_this_round = len(malicious_updates)
        args.malicious_attackers_this_round = malicious_attackers_this_round
        if args.attack == 'non_attack':
            malicious_attackers_this_round = 0
        
        print('attack numbers = ' + str(malicious_attackers_this_round))
        local_updates = malicious_updates + local_updates
        # gt attack ratio
        if args.num_attackers > 0:
            gt_attack_ratio = gt_attack_cnt / args.num_selected_users
            print('current iteration attack ratio: '+str(gt_attack_ratio))
            overall_attack_ratio.append(gt_attack_ratio)

        train_loss = sum(local_losses) / args.num_selected_users

        

        ################## <<< Attack Point 2: local model poisoning attacks
        if malicious_attackers_this_round != 0:
            local_updates = attack_method(local_updates, args, malicious_attackers_this_round)
        
        ## robust/non-robust global aggregation
        if args.attack:
            print('attack:' + args.attack)
        else:
            print('attack: None')

        if args.defend:
            print('defend:' + args.defend)
        else:
            print('defend: None')

        if args.defend == 'multi_krum':
            aggregate_model, _ = multi_krum(local_updates, multi_k=True)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'krum':
            aggregate_model, _ = multi_krum(local_updates, multi_k=False)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'bulyan':
            aggregate_model, _ = bulyan(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'tr_mean':
            aggregate_model = tr_mean(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'sparsefed':
            if t > 0:
                global_model, momentum, error = sparsefed(local_updates, global_model, args, momentum, error)
            else:
                global_model, momentum, error = sparsefed(local_updates, global_model, args)

        elif args.defend == 'signguard':
            global_model = signguard(local_updates, global_model, args)
        
        elif args.defend == 'dnc':
            global_model = dnc(local_updates, global_model, args)

        elif args.defend == 'lasa':
            global_model = lasa(local_updates, global_model, args)

        elif args.defend == 'geomed':
            global_model = geomed(local_updates, global_model, args)

        elif args.defend == 'fedavg':
            global_model = average(global_model, local_updates) # just fedavg

        ## test global model on server side
        net_glob.load_state_dict(global_model)
        with torch.no_grad():
            test_acc, _ = test_img(net_glob, dataset_test, args)

            with open(args.exp_record, 'a') as f:
                f.write('At round %d: the global model accuracy is %.5f' % (t, test_acc) + '\n')

                if t == args.round - 1:
                    f.write('-----' + '\n')
        print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
              format(t, train_loss, test_acc))
        
        if best_test_accuracy < test_acc:
            best_test_accuracy = test_acc

        if math.isnan(train_loss) or train_loss > 1e8 or t == args.round - 1:
            t2 = time.time()
            hours, rem = divmod(t2-t1, 3600)
            minutes, seconds = divmod(rem, 60)
            print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            print("best test accuracy ", best_test_accuracy)
            if len(overall_attack_ratio) > 0:
                print("overall poisoned ratio ", str(np.average(overall_attack_ratio)))
                return best_test_accuracy, np.average(overall_attack_ratio)
            else:
                return best_test_accuracy, 0
