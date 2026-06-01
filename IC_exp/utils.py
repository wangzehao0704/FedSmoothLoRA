#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import List, Dict, Union

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import *
import math

DATASET_NUM_MAP = {
    'cifar100': 100,
    'cifar10': 10,
    'food': 101,
    'flowers': 102,
    'cars': 196,
    'pets': 37,
    'aircraft': 100
}
def linear_zeta_schedule(current_round, total_rounds, initial_zeta=1, min_zeta=0):
    """
    Compute the Round-Matching coefficient zeta based on a linear schedule.

    The coefficient zeta controls the contribution of the Round-Matching matrix
    W_{c,rm}^{t} to the local LoRA initialization in FedSmoothLoRA.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_zeta: The initial value of zeta at round 0.
    :param min_zeta: The minimum value of zeta at the final round.
    :return: The zeta value for the current round.
    """
    if current_round > total_rounds:
        raise ValueError("current_round cannot exceed total_rounds.")

    zeta = initial_zeta - (current_round / total_rounds) * (initial_zeta - min_zeta)
    return zeta
def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr
def cosine_zeta_schedule(current_round, total_rounds, initial_zeta=1, min_zeta=0):
    """
    Compute the Round-Matching coefficient zeta based on a cosine schedule.

    Used by the ``decay`` mode of zeta in FedSmoothLoRA. Under heterogeneous
    (Non-IID) client distributions, the stabilizing effect of the
    Round-Matching matrix W_{c,rm}^{t} is preserved in early rounds while its
    influence is gradually reduced to avoid over-constraining heterogeneous
    local updates.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_zeta: The initial value of zeta at round 0.
    :param min_zeta: The minimum value of zeta at the final round.
    :return: The zeta value for the current round.
    """
    zeta = min_zeta + 0.5 * (initial_zeta - min_zeta) * (1 + math.cos(math.pi * current_round / total_rounds))
    return zeta
def matrix_truncated_svd_every_layer(A: Union[List[Dict], Dict], rank: int, niter: int, lowrank: bool = False,
                                     device='cuda') -> Union[List[Dict], Dict]:
    """
    Performs truncated Singular Value Decomposition (SVD) for each layer's weight matrix in the input dictionary `A`.
    If `A` is a list of dictionaries, it applies SVD for each layer. Otherwise, it applies SVD for the single dictionary.

    Args:
        A (Union[List[Dict], Dict]): The dictionary (or list of dictionaries) containing the weight matrices to apply SVD on.
        rank (int): The rank for the truncated SVD.
        niter (int): Number of iterations for `torch.svd_lowrank`.
        lowrank (bool, optional): Whether to use low-rank SVD. Defaults to False.
        device (str, optional): The device to move tensors to (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        Union[List[Dict], Dict]: A dictionary (or list of dictionaries) containing the truncated SVD results.
    """
    # If A is not a list, convert it to a list
    w_is_list = True
    if not isinstance(A, list):
        A = [A]
        w_is_list = False

    # Initialize the result as a list of empty dictionaries
    w_svd = [{} for _ in range(len(A))]

    for i in range(len(A)):
        for key in tqdm(A[i].keys(), desc=f"Processing layer {i + 1} keys", total=len(A[i].keys())):
            if lowrank:
                u, s, v = torch.svd_lowrank(A[i][key].to(device), min(4 * rank, min(A[i][key].shape)), niter)
            else:
                u, s, v = torch.svd(A[i][key].to(device))
            u = u[:, :rank]
            s = s[:rank]
            v = v.T[:rank, :]
            sqrt_s = torch.sqrt(s)

            u = u @ torch.diag(sqrt_s)
            v = torch.diag(sqrt_s) @ v
            w_svd[i][key.replace('lora_BA', 'lora_B')] = u.to(device)
            w_svd[i][key.replace('lora_BA', 'lora_A')] = v.to(device)

    # If there was only one element, return the single dictionary
    if not w_is_list:
        return w_svd[0]

    return w_svd



def matrix_addition_every_layer(A: Union[List[Dict], Dict], B: Union[List[Dict], Dict], device) -> Union[
    List[Dict], Dict]:
    """
    Performs element-wise addition between two weight dictionaries (or lists of dictionaries).
    For each layer, it adds the corresponding matrices in `A` and `B` and returns the result.
    If input is a list of dictionaries, addition is done for each layer.
    If input is a single dictionary, addition is done for that dictionary.

    Args:
        A (Union[List[Dict], Dict]): The first dictionary or list of dictionaries containing the weight matrices (被加数).
        B (Union[List[Dict], Dict]): The second dictionary or list of dictionaries containing the weight matrices (加数).
        device: The device to move tensors to (e.g., 'cuda' or 'cpu').

    Returns:
        Union[List[Dict], Dict]: A dictionary (or list of dictionaries) containing the result of the addition.
    """
    # If A or B is not a list, convert them to lists
    w_is_list = True

    if not isinstance(A, list):
        A = [A]
        w_is_list = False
    if not isinstance(B, list):
        B = [B]
        w_is_list = False

    # Initialize the result as a list of empty dictionaries
    result = [{} for _ in range(len(A))]

    for i in range(len(A)):
        for key in A[i].keys():
            if key in B[i]:
                # Perform element-wise addition
                result[i][key] = A[i][key].to(device) + B[i][key].to(device)

    # If there was only one element, return the single dictionary
    if not w_is_list:
        return result[0]

    return result


def matrix_subtraction_every_layer(A: Union[List[Dict], Dict], B: Union[List[Dict], Dict],  device,beta = 1.0) -> \
Union[List[Dict], Dict]:
    """
    Performs element-wise subtraction between two weight dictionaries (or lists of dictionaries).
    For each layer, it subtracts the corresponding matrices in `A` and `B` and returns the result.
    If input is a list of dictionaries, subtraction is done for each layer.
    If input is a single dictionary, subtraction is done for that dictionary.

    Args:
        A (Union[List[Dict], Dict]): The first dictionary or list of dictionaries containing the weight matrices (被减数).
        B (Union[List[Dict], Dict]): The second dictionary or list of dictionaries containing the weight matrices (减数).
        device: The device to move tensors to (e.g., 'cuda' or 'cpu').

    Returns:
        Union[List[Dict], Dict]: A dictionary (or list of dictionaries) containing the result of the subtraction.
    """
    # If A or B is not a list, convert them to lists
    w_is_list = True

    if not isinstance(A, list):
        A = [A]
        w_is_list = False
    if not isinstance(B, list):
        B = [B]
        w_is_list = False
    # Initialize the result as a list of empty dictionaries
    result = [{} for _ in range(len(A))]

    for i in range(len(A)):
        for key in A[i].keys():
            if key in B[i]:
                # Perform element-wise subtraction and move tensors to the specified device
                result[i][key] = beta *(A[i][key].to(device) - B[i][key].to(device))

    # If the input was a single dictionary, return just the first result
    if not w_is_list:
        return result[0]

    return result


def matrix_multiply_every_layer(w: Union[List[Dict], Dict], device):
    """
    Performs matrix multiplication for every layer's weight dictionary in the input `w`.
    If `w` is a list, it performs matrix multiplication for each layer in the list of dictionaries.
    If `w` is a single dictionary, it performs the multiplication for that dictionary.

    Returns a list of dictionaries (or a single dictionary if `w` was not a list).
    """
    # If `w` is not a list, convert it to a list with a single element
    w_is_list = True
    if not isinstance(w, list):
        w = [w]
        w_is_list = False

    w_ba = [{} for _ in range(len(w))]  # Create an empty dictionary for each layer

    for i in range(len(w)):
        for key in w[i].keys():
            if 'lora_A' in key:
                # Perform matrix multiplication: replace 'lora_A' with 'lora_BA'
                new_key = key.replace('lora_A', 'lora_BA')
                w_ba[i][new_key] = torch.matmul(w[i][key.replace('lora_A', 'lora_B')].to(device), w[i][key].to(device))

    # If `w` was a single dictionary, return just the first dictionary in `w_ba`
    if not w_is_list:
        return w_ba[0]

    return w_ba


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # datasets.
    dataset_mapping = {
        'cifar100': lambda: (
            datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=data_transforms),
            datasets.CIFAR100('./data/cifar100/', train=False, download=True, transform=data_transforms)
        ),
        'cifar10': lambda: (
            datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=data_transforms),
            datasets.CIFAR10('./data/cifar10/', train=False, download=True, transform=data_transforms)
        ),
        'food': lambda: (
            datasets.Food101(root='./data/food101', split='train', download=True, transform=data_transforms),
            datasets.Food101(root='./data/food101', split='test', download=True, transform=data_transforms)
        ),
        'flowers': lambda: (
            datasets.Flowers102(root='./data/flowers102', split='train', download=True, transform=data_transforms),
            datasets.Flowers102(root='./data/flowers102', split='test', download=True, transform=data_transforms)
        ),
        'cars': lambda: (
            datasets.StanfordCars(root='./data/stanford_cars', split='train', download=True, transform=data_transforms),
            datasets.StanfordCars(root='./data/stanford_cars', split='test', download=True, transform=data_transforms)
        ),
        'pets': lambda: (
            datasets.OxfordIIITPet(root='./data/oxford_iiit_pet', split='train', download=True, transform=data_transforms),
            datasets.OxfordIIITPet(root='./data/oxford_iiit_pet', split='test', download=True, transform=data_transforms)
        ),
        'aircraft': lambda: (
            datasets.FGVCAircraft(root='./data/fgvc_aircraft', split='train', download=True, transform=data_transforms),
            datasets.FGVCAircraft(root='./data/fgvc_aircraft', split='test', download=True, transform=data_transforms)
        )
    }

    if args.dataset not in dataset_mapping:
        raise ValueError("Dataset not supported.")

    train_dataset, test_dataset = dataset_mapping[args.dataset]()

    user_groups = (
        dataset_iid(train_dataset, args.num_users) if args.iid else dataset_dir(train_dataset, args.num_users)
    )

    return train_dataset, test_dataset, user_groups


def average_weights(w, count_list, idxs_users):
    """
    Returns the weighted average of the weights based on count_list.

    :param w: List of weight dictionaries.
    :param count_list: List containing the count for each weight in w.
    :return: Weighted average of weights.
    """
    temp_count_list = [count_list[i] for i in idxs_users]
    temp_w = [w[i] for i in idxs_users]
    total_count = sum(temp_count_list)
    keys = list(temp_w[0].keys())
    w_avg = {key: 0 for key in keys}

    for i, weights in enumerate(temp_w):
        weight_factor = count_list[i] / total_count
        for key in keys:
            w_avg[key] += weights[key] * weight_factor

    return w_avg



def merge_ba_weights(w):
    """
    Returns the average of the weights.
    """

    w_ba = [{} for _ in range(len(w))]
    for i in range(0, len(w)):
        for key in w[i].keys():
            if 'lora_A' in key:
                w_ba[i][key.replace('lora_A', 'lora_BA')] = w[i][key.replace('lora_A','lora_B')] @w[i][key]
    return w_ba


def svd_ba(w,rank):
    """
    Returns the average of the weights.
    """

    w_svd = {}
    for key in w.keys():
        u, s, v = torch.svd(w[key])
        u = u[:, :rank]
        s = s[:rank]
        v = v.T[:rank, :]
        w_svd[key.replace('lora_BA', 'lora_B')] = u @ torch.diag(s)
        w_svd[key.replace('lora_BA', 'lora_A')] = v
    return w_svd
def svd_ba_fast(w,rank,target):
    """
    Returns the average of the weights.
    """

    w_svd = {}
    for key in w.keys():
        print(w[key].size())
        u, s, v = torch.svd_lowrank(w[key].cuda(),target,10)
        u = u[:, :rank]
        s = s[:rank]
        v = v.T[:rank, :]
        w_svd[key.replace('lora_BA', 'lora_B')] = u @ torch.diag(s)
        w_svd[key.replace('lora_BA', 'lora_A')] = v
    return w_svd
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
