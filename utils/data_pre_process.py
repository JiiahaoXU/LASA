import numpy as np
import os
import dill
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import defaultdict
import json


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/dataset/femnist/train",
                                                                                 "./data/dataset/femnist/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class custom_subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The subset Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

# split for federated settings
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


################################### data setup ########################################
def load_partition(args):
    dict_users = []
    # read dataset
    if args.dataset == 'mnist':
        path = './data/dataset/mnist'
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
        args.num_classes = 10
        # split training dataset (iid or non-iid)
        # remove dict_users.pik at the first time. 

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                dict_users = noniid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)

    elif args.dataset == 'fmnist':
        path = './data/dataset/fmnist'
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
        args.num_classes = 10

        # pik_name = args.config_name.split('/')[-1].split('.')[0]
        # pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        pik_path = os.path.join(path,'fmnist_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                dict_users = noniid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)

    elif args.dataset == 'noniidfmnist':
        path = './data/dataset/fmnist'
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
        args.num_classes = 10

        # pik_name = args.config_name.split('/')[-1].split('.')[0]
        # pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        pik_path = os.path.join(path,'fmnist_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            
            dict_users = noniid(dataset_train, args.num_users)
            if args.freeze_datasplit:
                with open(pik_path, 'wb') as f: 
                    dill.dump(dict_users, f)
            
    elif args.dataset == 'svhn':
        path = './data/dataset/svhn'
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])
        dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
        # dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
        # dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
        dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
        args.num_classes = 10

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                exit('Error: only consider IID setting in SVHN')

    elif args.dataset == 'emnist':
        path = './data/dataset/emnist'
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
        dataset_train = datasets.EMNIST(path, split='balanced', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST(path, split='balanced', train=False, download=True, transform=trans_emnist)
        args.num_classes = 10

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                exit('Error: only consider IID setting in emnist')

    elif args.dataset == 'cifar':
        path = './data/dataset/cifar'
        dataset_train = datasets.CIFAR10(path, train=True, download=True, 
                                        transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))]))
        dataset_test = datasets.CIFAR10(path, train=False, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))]))
        dataset_train.targets, dataset_test.targets = torch.LongTensor(dataset_train.targets), torch.LongTensor(
            dataset_test.targets)
        args.num_classes = 10

        def distribute_data(dataset, args, n_classes=10):
            # logging.info(dataset.targets)
            # logging.info(dataset.classes)
            class_per_agent = n_classes

            if args.num_users == 1:
                return {0: range(len(dataset))}

            def chunker_list(seq, size):
                return [seq[i::size] for i in range(size)]

            # sort labels
            labels_sorted = torch.tensor(dataset.targets).sort()
            # print(labels_sorted)
            # create a list of pairs (index, label), i.e., at index we have an instance of  label
            class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
            # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
            labels_dict = defaultdict(list)
            for k, v in class_by_labels:
                labels_dict[k].append(v)

            # split indexes to shards
            shard_size = len(dataset) // (args.num_users * class_per_agent)
            slice_size = (len(dataset) // n_classes) // shard_size
            for k, v in labels_dict.items():
                labels_dict[k] = chunker_list(v, slice_size)
            import copy
            hey = copy.deepcopy(labels_dict)
            # distribute shards to users
            dict_users = defaultdict(list)
            for user_idx in range(args.num_users):
                class_ctr = 0
                for j in range(0, n_classes):
                    if class_ctr == class_per_agent:
                        break
                    elif len(labels_dict[j]) > 0:
                        dict_users[user_idx] += labels_dict[j][0]
                        del labels_dict[j % n_classes][0]
                        class_ctr += 1
                np.random.shuffle(dict_users[user_idx])

            return dict_users
        
        dict_users = distribute_data(dataset_train, args)
    
    elif args.dataset == 'noniidcifar':
        path = './data/dataset/cifar'
        dataset_train = datasets.CIFAR10(path, train=True, download=True, 
                                        transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))]))
        dataset_test = datasets.CIFAR10(path, train=False, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))]))
        dataset_train.targets, dataset_test.targets = torch.LongTensor(dataset_train.targets), torch.LongTensor(
            dataset_test.targets)
        args.num_classes = 10

        def distribute_data_dirichlet(dataset, args):
            # sort labels
            labels_sorted = dataset.targets.sort()
            # create a list of pairs (index, label), i.e., at index we have an instance of  label
            class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
            labels_dict = defaultdict(list)

            for k, v in class_by_labels:
                labels_dict[k].append(v)
            # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
            N = len(labels_sorted[1])
            K = len(labels_dict)
            # logging.info((N, K))
            client_num = args.num_users

            min_size = 0
            while min_size < 10:
                idx_batch = [[] for _ in range(client_num)]
                for k in labels_dict:
                    idx_k = labels_dict[k]

                    # get a list of batch indexes which are belong to label k
                    np.random.shuffle(idx_k)
                    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
                    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
                    proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

                    # get the index in idx_k according to the dirichlet distribution
                    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                    # generate the batch list for each client
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            # distribute data to users
            dict_users = defaultdict(list)
            for user_idx in range(args.num_users):
                dict_users[user_idx] = idx_batch[user_idx]
                np.random.shuffle(dict_users[user_idx])

            return dict_users

        dict_users = distribute_data_dirichlet(dataset_train, args)

    elif args.dataset == 'cifar100':
        path = './data/dataset/cifar100'
        dataset_train = datasets.CIFAR100(path, train=True, download=True, 
                                        transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])]))
        dataset_test = datasets.CIFAR100(path, train=False, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])]))
        dataset_train.targets, dataset_test.targets = torch.LongTensor(dataset_train.targets), torch.LongTensor(
            dataset_test.targets)
        args.num_classes = 100

        def distribute_data(dataset, args, n_classes=100):
            # logging.info(dataset.targets)
            # logging.info(dataset.classes)
            class_per_agent = args.num_classes

            if args.num_users == 1:
                return {0: range(len(dataset))}

            def chunker_list(seq, size):
                return [seq[i::size] for i in range(size)]

            # sort labels
            labels_sorted = torch.tensor(dataset.targets).sort()
            # print(labels_sorted)
            # create a list of pairs (index, label), i.e., at index we have an instance of  label
            class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
            # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
            labels_dict = defaultdict(list)
            for k, v in class_by_labels:
                labels_dict[k].append(v)

            # split indexes to shards
            shard_size = len(dataset) // (args.num_users * class_per_agent)
            slice_size = (len(dataset) // n_classes) // shard_size
            for k, v in labels_dict.items():
                labels_dict[k] = chunker_list(v, slice_size)
            import copy
            hey = copy.deepcopy(labels_dict)
            # distribute shards to users
            dict_users = defaultdict(list)
            for user_idx in range(args.num_users):
                class_ctr = 0
                for j in range(0, n_classes):
                    if class_ctr == class_per_agent:
                        break
                    elif len(labels_dict[j]) > 0:
                        dict_users[user_idx] += labels_dict[j][0]
                        del labels_dict[j % n_classes][0]
                        class_ctr += 1
                np.random.shuffle(dict_users[user_idx])

            return dict_users
        
        dict_users = distribute_data(dataset_train, args)


    elif args.dataset == 'noniidcifar100':
        path = './data/dataset/cifar100'
        dataset_train = datasets.CIFAR100(path, train=True, download=True, 
                                        transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])]))
        dataset_test = datasets.CIFAR100(path, train=False, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])]))
        dataset_train.targets, dataset_test.targets = torch.LongTensor(dataset_train.targets), torch.LongTensor(
            dataset_test.targets)
        args.num_classes = 100

        def distribute_data_dirichlet(dataset, args):
            # sort labels
            labels_sorted = dataset.targets.sort()
            # create a list of pairs (index, label), i.e., at index we have an instance of  label
            class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
            labels_dict = defaultdict(list)

            for k, v in class_by_labels:
                labels_dict[k].append(v)
            # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
            N = len(labels_sorted[1])
            K = len(labels_dict)
            # logging.info((N, K))
            client_num = args.num_users

            min_size = 0
            while min_size < 10:
                idx_batch = [[] for _ in range(client_num)]
                for k in labels_dict:
                    idx_k = labels_dict[k]

                    # get a list of batch indexes which are belong to label k
                    np.random.shuffle(idx_k)
                    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
                    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
                    proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

                    # get the index in idx_k according to the dirichlet distribution
                    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                    # generate the batch list for each client
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            # distribute data to users
            dict_users = defaultdict(list)
            for user_idx in range(args.num_users):
                dict_users[user_idx] = idx_batch[user_idx]
                np.random.shuffle(dict_users[user_idx])

            return dict_users

        dict_users = distribute_data_dirichlet(dataset_train, args)


    elif args.dataset == 'shakespeare':
        dataset_train = torch.load('./data/dataset/shakespeare/train.pt')
        dataset_test = torch.load('./data/dataset/shakespeare/test.pt')
        dict_users = torch.load('./data/dataset/shakespeare/dict_users.pt')
        # dict_users = dataset_train.get_client_dic()
        # print(len(dataset_train), len(dataset_test))
        args.num_users = len(dict_users)
        if args.iid:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        # dict_users = noniid(dataset_train, args.num_users)
        # print(dict_users)
        args.num_users = len(dict_users)

        if args.iid:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    
    ## extract 10% data from test set for validation, and the rest for testing
    print("Creating validation dataset from testing dataset...")
    dataset_test, dataset_val = torch.utils.data.random_split(dataset_test, [len(dataset_test)-int(0.1 * len(dataset_test)), int(0.1 * len(dataset_test))])
    ## generate a public dataset for DP-topk purpose from validation set
    dataset_test, dataset_val = dataset_test.dataset, dataset_val.dataset
    # print("Creating public dataset...")
    # dataset_public = public_iid(dataset_val, args) # make sure public set has every class
    ## make sure experiments with different sizes of public dataset use the same testing data and training data

    # return args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users
    return args, dataset_train, dataset_test, dataset_val, None, dict_users

###################### utils #################################################
## IID assign data samples for num_users (mnist, svhn, fmnist, emnist, cifar)
def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    print("Assigning training data samples (iid)")
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

## IID assign data samples for num_users (mnist, emnist, cifar); each user only has n(default:two) classes of data
# def noniid(dataset, num_users, class_num=2):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: each user only has two classes of data
#     """
#     print("Assigning training data samples (non-iid)")
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, class_num, replace=False))
#         if num_users <= num_shards:
#             idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

def noniid(dataset, num_users, class_num=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: each user only has two classes of data
    """
    print("Assigning training data samples (non-iid)")
        # 60,000 training imgs
    s = 0.1
    num_per_user = int(50000/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    # print(np.array(dataset.targets))
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = num_users * 2, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    # np.random.seed(111)
    for i in range(num_users):
        # allocate iid idxs
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid,replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        # print(idx_shard, i)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])
    return dict_users

## generate a iid public dataset from dataset. 
def public_iid(dataset, args):
    """
    Sample I.I.D. public data from fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if args.dataset == 'fmnist' or args.dataset == 'mnist':
        labels = dataset.train_labels.numpy()
    elif args.dataset == 'cifar':
        labels = np.array(dataset.targets)
    else:
        labels = dataset.labels
    pub_set_idx = set()
    if args.pub_set > 0:
        for i in list(set(labels)):
            pub_set_idx.update(
                set(
                np.random.choice(np.where(labels==i)[0],
                                          int(args.pub_set/len(list(set(labels)))), 
                                 replace=False)
                )
                )
    # test_set_idx = set(np.arange(len(labels)))
    # test_set_idx= test_set_idx.difference(val_set_idx)
    return DatasetSplit(dataset, pub_set_idx)

def sample_dirichlet_train_data(dataset, args, no_participants, alpha=0.9):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if ind in args.poison_images or ind in args.poison_images_test:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])
    per_participant_list = {}
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            if user in per_participant_list:
                per_participant_list[user].extend(sampled_list)
            else:
                per_participant_list[user] = sampled_list
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

    return per_participant_list
