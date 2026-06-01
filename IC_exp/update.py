#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
import copy
from typing import Tuple, List, Dict
from tqdm import tqdm
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook
class LocalUpdate(object):
    def __init__(self, args, dataset, lr, idx, idxs, logger):
        self.args = args
        self.lr = lr
        self.idx = idx
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def estimate_gradient(self, model, sample_size = 128 , bsz = 4, add_param=None):
        r"""
        Estimate the gradient of the model on the given dataset
        """
        if add_param!=None:
            for name, param in model.named_parameters():
                if 'base_layer.weight' in name:
                    if self.args.use_rslora:
                        param.data = param.data + (self.args.lora_alpha / math.sqrt(self.args.lora_r)) * add_param[
                            name.replace('base_layer.weight', 'lora_BA.weight')].to(self.device)
                    else:
                        param.data = param.data + (self.args.lora_alpha / self.args.lora_r) * add_param[
                            name.replace('base_layer.weight', 'lora_BA.weight')].to(self.device)

        print("Estimating gradient")
        model = model.unload()
        model.train()
        named_grads = {}
        hooks = []
        for name, param in model.named_parameters():
            param.requires_grad_()
            hook = param.register_hook(get_record_gradient_hook(model, named_grads))
            hooks.append(hook)
        num = 0
        criterion = nn.CrossEntropyLoss().to(self.device)
        trainloader = copy.deepcopy(self.trainloader)

        for images, labels in tqdm(trainloader, desc="Estimating gradient"):
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs = model(images)
            num += 1
            # batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = criterion(log_probs, labels)
            loss.backward()
            get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
            # make sure the gradient is cleared
            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad = None
            if num == sample_size // bsz:
                break
        for n, g in named_grads.items():
            named_grads[n] /= num
        for hook in hooks:
            hook.remove()
        torch.cuda.empty_cache()
        for name, param in model.named_parameters():
            param.requires_grad=False

        return named_grads
    def get_ga_dict(self,local_named_grads, gamma):
        local_grad = {}

        for name, param in local_named_grads.items():
            if 'qkv.weight' in name:
                name = name.replace('blocks', 'base_model.model.blocks')
                name = name.replace('qkv.weight', 'qkv.lora_BA.weight')
                local_grad[name] = param
            if 'head.weight' in name:
                name = name.replace('head', 'base_model.model.head')
                name = name.replace('head.weight', 'head.lora_BA.weight')
                local_grad[name] = param
        local_grad_svd = {}
        for name, param in local_grad.items():
            U, S, V = torch.svd(param.to(self.device))
            V = V.T
            B = U[:, self.args.lora_r: 2 * self.args.lora_r]
            A = V[:self.args.lora_r, :]
            if gamma!=0:
                p, q = param.shape
                B = B * p ** 0.25 / gamma ** 0.5
                A = A * p ** 0.25 / gamma ** 0.5
            local_grad_svd[name.replace('lora_BA', 'lora_B')] = B
            local_grad_svd[name.replace('lora_BA', 'lora_A')] = A
        return local_grad_svd
    def get_pissa_dict(self,local_named_grads, gamma):
        local_grad = {}

        for name, param in local_named_grads.items():
            if 'qkv.weight' in name:
                name = name.replace('blocks', 'base_model.model.blocks')
                name = name.replace('qkv.weight', 'qkv.lora_BA.weight')
                local_grad[name] = param
            if 'head.weight' in name:
                name = name.replace('head', 'base_model.model.head')
                name = name.replace('head.weight', 'head.lora_BA.weight')
                local_grad[name] = param
        local_grad_svd = {}
        for name, param in local_grad.items():
            rank = self.args.lora_r
            u, s, v = torch.svd(param.to(self.device))
            u = u[:, :rank]
            s = s[:rank]
            v = v.T[:rank, :]
            sqrt_s = torch.sqrt(s)

            u = u @ torch.diag(sqrt_s)
            v = torch.diag(sqrt_s) @ v
            # if gamma!=0:
            #     p, q = param.shape
            #     B = B * p ** 0.25 / gamma ** 0.5
            #     A = A * p ** 0.25 / gamma ** 0.5
            local_grad_svd[name.replace('lora_BA', 'lora_B')] = u
            local_grad_svd[name.replace('lora_BA', 'lora_A')] = v
        return local_grad_svd
    def update_weights(self, model, global_round, client_step):

        # Set mode to train model
        model.train()
        epoch_loss = []
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=self.lr,
                                        momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(trainable_params, lr=self.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()) )
                self.logger.add_scalar('train_loss/client_{}'.format(self.idx), loss.item(), client_step[self.idx])
                client_step[self.idx] = client_step[self.idx] + 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return copy.deepcopy(get_peft_model_state_dict(model)), sum(epoch_loss) / len(epoch_loss), client_step

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    del testloader
    accuracy = correct/total
    return accuracy, loss
