import torch
from torch import nn
import copy


class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        if args.data_type == 'image':
            self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        elif args.data_type == 'text':
            self.loss_func = nn.CrossEntropyLoss().to(self.args.device)

    def sgd(self, net, samples, labels):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        net.zero_grad()
        log_probs = net(samples)
        loss = self.loss_func(log_probs, labels)
        loss.backward()
        # update
        optimizer.step()
        w_new = copy.deepcopy(net.state_dict())
        return w_new, loss.item()
    
  
    
    def local_sgd(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

    def local_sgd_mome(self, net, ldr_train, topk_model=None, mask=None, attack_flag=None, attack_method=None, num_of_label=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                if attack_flag and attack_method =='label_flip':
                    labels = num_of_label - labels
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                if mask is not None:
                    for name, weight in net.named_parameters():

                        if name in mask:
                            weight.grad.data = weight.grad.data * mask[name]

                        
                optimizer.step()

                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)
    
