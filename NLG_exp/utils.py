import random
import time
import tracemalloc
import math
import torch
import typing as tp
import pandas as pd
import wandb
from torch.optim import SGD
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
)
from datasets import Dataset
from logTrainer import LogTrainer, LogTrainerFedProx, LogTrainerSCAFFOLD
import logging
from copy import deepcopy
from transformers.trainer_utils import PredictionOutput

import torch
from typing import List, Dict, Union

log = logging.getLogger(__name__)
def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
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

import torch
from tqdm import tqdm
from typing import Dict, Union


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
                u, s, v = torch.svd_lowrank(A[i][key].to('cuda'), min(4 * rank, min(A[i][key].shape)), niter)
            else:
                u, s, v = torch.svd(A[i][key].to('cuda'))
            u = u[:, :rank]
            s = s[:rank]
            v = v.T[:rank, :]
            w_svd[i][key.replace('lora_BA', 'lora_B')] = (u @ torch.diag(s)).to(device)
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
def get_proxy_dict(fed_alg, fedopt_tau, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fedopt_tau**2
    elif fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict


def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round, **kwargs):

    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])

    if kwargs.get("fed_alg", 'fedavg') == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = kwargs.get("auxiliary_info", None)
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]).cpu()
            global_auxiliary[key] += delta_auxiliary / kwargs.get("num_clients", None)
    elif kwargs.get("fed_alg", 'fedavg') == 'fedadam':
        for key, param in kwargs.get("opt_proxy_dict", None).items():
            delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(
                clients_this_round)
            kwargs.get("proxy_dict", None)[key] = kwargs.get("fedopt_beta1", None) * kwargs.get("proxy_dict", None)[key].cpu() + (
                        1 - kwargs.get("fedopt_beta1", None)) * delta_w.cpu() if kwargs.get("round_idx", None) > 0 else delta_w
            kwargs.get("opt_proxy_dict", None)[key] = kwargs.get("fedopt_beta2", None) * param.cpu() + (1 - kwargs.get("fedopt_beta2", None)) * torch.square(
                kwargs.get("proxy_dict", None)[key].cpu())
            global_dict[key] += kwargs.get("fedopt_eta", None) * torch.div(kwargs.get("proxy_dict", None)[key].cpu(),
                                                                torch.sqrt(kwargs.get("opt_proxy_dict", None)[key].cpu()) + kwargs.get("fedopt_tau", None))

    elif kwargs.get("fed_alg", 'fedavg') == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            kwargs.get("proxy_dict", None)[key] = kwargs.get("fedopt_beta1", None) * kwargs.get("proxy_dict", None)[key] + (1 - kwargs.get("fedopt_beta1", None)) * delta_w.cpu() if kwargs.get("round_idx", None) > 0 else delta_w.cpu()
            global_dict[key] = global_dict[key].cpu() + kwargs.get("proxy_dict", None)[key].cpu()

    elif kwargs.get("fed_alg", 'fedavg') == 'fedadagrad':
        for key, param in kwargs.get("opt_proxy_dict", None).items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            kwargs.get("proxy_dict", None)[key] = delta_w.cpu()
            kwargs.get("opt_proxy_dict", None)[key] = param.cpu() + torch.square(kwargs.get("proxy_dict", None)[key].cpu())
            global_dict[key] += kwargs.get("fedopt_eta", None) * torch.div(kwargs.get("proxy_dict", None)[key].cpu(), torch.sqrt(kwargs.get("opt_proxy_dict", None)[key].cpu())+kwargs.get("fedopt_tau", None))

    elif kwargs.get("fed_alg", 'fedavg') == 'fedyogi':
        for key, param in kwargs.get("opt_proxy_dict", None).items():
            delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
            kwargs.get("proxy_dict", None)[key] = kwargs.get("fedopt_beta1", None) * kwargs.get("proxy_dict", None)[key].cpu()+ (1 - kwargs.get("fedopt_beta1", None)) * delta_w.cpu() if kwargs.get("round_idx", None) > 0 else delta_w.cpu()
            delta_square = torch.square(kwargs.get("proxy_dict", None)[key].cpu())
            kwargs.get("opt_proxy_dict", None)[key] = param.cpu() - (1-kwargs.get("fedopt_beta2", None))*delta_square*torch.sign(param.cpu() - delta_square)
            global_dict[key] += kwargs.get("fedopt_eta", None) * torch.div(kwargs.get("proxy_dict", None)[key].cpu(), torch.sqrt(kwargs.get("opt_proxy_dict", None)[key].cpu())+kwargs.get("fedopt_tau", None))


    else:
        for key in global_dict.keys():
            global_dict[key] = sum(
                [local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in
                 clients_this_round])
    return global_dict



def global_stack(global_dict, local_dict_list, sample_num_list, clients_this_round, target_key, use_weights=True,
                 device='cuda:0'):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    new_global_dict = {}

    for key in global_dict.keys():
        if target_key in key:
            stacked_params_list = []
            for client in clients_this_round:
                param = local_dict_list[client][key].to(device)

                num_samples = sample_num_list[client]/sample_this_round

                stacked_params_list.append(param*num_samples if use_weights else param)
            if target_key == 'lora_A':
                stacked_params = torch.cat(stacked_params_list, dim=0)
            else:
                stacked_params = torch.cat(stacked_params_list, dim=1)
            new_global_dict[key] = stacked_params

    return new_global_dict


def get_clients_this_round(num_clients, round):
    random.seed(round)
    clients_this_round = sorted(random.sample(range(num_clients), num_clients))
    return clients_this_round
def get_dataset_this_round(dataset, round, batch_size, local_steps):
    num2sample = batch_size * local_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round


import random
import numpy as np
from torch.utils.data import Subset


def split_dataset(num_clients, seed, dataset, dataset_name=None, alpha=1.0, non_iid=False):
    dataset = dataset.shuffle(seed=seed)  # Shuffle the dataset
    local_datasets = []
    if non_iid:
        if 'aya' in dataset_name:
            train_labels = {}
            count = 0
            for i in tqdm(dataset):
                train_labels[count] = i['language']
                count += 1
            n_classes = 44
            label_distribution = np.random.dirichlet([alpha] * num_clients, 44)
            class_idcs = {}

            for idx, label in train_labels.items():
                if label not in class_idcs:
                    class_idcs[label] = []
                class_idcs[label].append(idx)


            client_idcs = [[] for _ in range(num_clients)]

            for k_idcs, fracs in zip(class_idcs.values(), label_distribution):
                if len(k_idcs) == 0:
                    continue

                split_indices = (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int)

                if len(k_idcs) <= 1 or (len(split_indices) > 0 and max(split_indices) >= len(k_idcs)):
                    print(f"Warning: Insufficient samples in k_idcs, assigning to first client.")
                    client_idcs[0].append(k_idcs)
                    continue

                for i, idcs in enumerate(np.split(k_idcs, split_indices)):
                    client_idcs[i].append(idcs)

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

            for i in range(num_clients):
                local_datasets.append(dataset.select(client_idcs[i]))
            import matplotlib.pyplot as plt
            from collections import Counter
            for client_id, client_indices in enumerate(client_idcs):
                client_languages = [train_labels[idx] for idx in client_indices]
                language_counts = Counter(client_languages)

                plt.figure(figsize=(8, 8))
                plt.pie(language_counts.values(), labels=language_counts.keys(), autopct='%1.1f%%', startangle=140)
                plt.title(f'Client {client_id} Language Distribution')
                plt.show()
        elif 'dolly' in dataset_name:
            train_labels = {}
            count = 0
            for i in tqdm(dataset):
                train_labels[count] = i['cate']
                count += 1
            n_classes = 8
            label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)
            class_idcs = {}

            for idx, label in train_labels.items():
                if label not in class_idcs:
                    class_idcs[label] = []
                class_idcs[label].append(idx)


            client_idcs = [[] for _ in range(num_clients)]

            for k_idcs, fracs in zip(class_idcs.values(), label_distribution):
                if len(k_idcs) == 0:
                    continue

                split_indices = (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int)

                if len(k_idcs) <= 1 or (len(split_indices) > 0 and max(split_indices) >= len(k_idcs)):
                    print(f"Warning: Insufficient samples in k_idcs, assigning to first client.")
                    client_idcs[0].append(k_idcs)
                    continue

                for i, idcs in enumerate(np.split(k_idcs, split_indices)):
                    client_idcs[i].append(idcs)

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

            for i in range(num_clients):
                local_datasets.append(dataset.select(client_idcs[i]))
            import matplotlib.pyplot as plt
            from collections import Counter
            for client_id, client_indices in enumerate(client_idcs):
                client_languages = [train_labels[idx] for idx in client_indices]
                language_counts = Counter(client_languages)
                import matplotlib.pyplot as plt
                plt.rcParams['font.family'] = 'Times New Roman'

                plt.figure(figsize=(8, 8))
                plt.pie(language_counts.values(), labels=language_counts.keys(), autopct='%1.1f%%', startangle=140)
                # plt.title(f'Distribution of instruction categories for Client {client_id}')
                plt.savefig(f'image_{client_id}')
                plt.show()
        else:
            train_labels = {}
            count = 0
            for i in tqdm(dataset):
                train_labels[count] = i['label']
                count+=1
            n_classes = max(train_labels.values()) + 1
            label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)
            class_idcs = [ [] for y in range(n_classes)]

            for idx, label in train_labels.items():
                class_idcs[label].append(idx)

            client_idcs = [[] for _ in range(num_clients)]
            for k_idcs, fracs in zip(class_idcs, label_distribution):

                for i, idcs in enumerate(np.split(k_idcs,
                                                  (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                          astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            for i in range(num_clients):
                local_datasets.append(dataset.select(client_idcs[i]))

    else:

        for i in range(num_clients):
            local_datasets.append(dataset.shard(num_clients, i))

    return local_datasets



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def seed_everything_npu(seed: int):
    import random, os
    import numpy as np
    import torch_npu

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def find_all_linear_modules(model) -> tp.List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def causalLMEncode(example, tokenizer, max_length=-1, ignore_masked_token=True):
    is_list_input = isinstance(example["x"], list)
    # Combine text and add EOS token
    combined_text = (
        [
            x + " " + y + tokenizer.eos_token
            for (x, y) in zip(example["x"], example["y"])
        ]
        if is_list_input
        else example["x"] + " " + example["y"] + tokenizer.eos_token
    )
    # Tokenize combined text
    encodings = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length if max_length != -1 else None,
    )
    # Calculate input text length in tokens
    input_text_length = (
        [
            len(tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
            for i in range(len(example["x"]))
        ]
        if is_list_input
        else len(tokenizer(example["x"], return_tensors="pt")["input_ids"][0])
    )
    if input_text_length[0] >= max_length:
        log.warning(
            f"Input text length >= max_length: {input_text_length} >= {max_length}. "
            "Consider increasing max_length to avoid truncation."
        )
    # Create labels
    labels = encodings["input_ids"].clone()
    if is_list_input:
        for i, l in enumerate(input_text_length):
            labels[i, :l] = -100
    else:
        labels[0, :input_text_length] = -100
    if ignore_masked_token:
        labels[encodings["attention_mask"] == 0] = -100
    # Update example dictionary
    results = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        # "input_text_length": input_text_length,
    }

    return results


def SeqToSeqEncode(example, tokenizer, max_length=None, ignore_masked_token=False):
    inputs = tokenizer(
        example["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example["y"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
        "decoder_attention_mask": outputs["attention_mask"],
    }

    if ignore_masked_token:
        results["labels"][outputs["attention_mask"] == 0] = -100

    return results


def preprocess_dataset(
    dataset: tp.Union[Dataset, tp.List[tp.Tuple[str, str]], tp.List[tp.Dict[str, str]]]
) -> Dataset:
    if isinstance(dataset, list) and isinstance(dataset[0], tuple):
        dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=["x", "y"]))
    elif isinstance(dataset, list) and isinstance(dataset[0], dict):
        dataset = Dataset.from_dict(
            {k: [dic[k] for dic in dataset] for k in dataset[0]}
        )
    elif isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset)
    elif isinstance(dataset, Dataset):
        pass
    else:
        raise ValueError("Wrong format")
    return dataset


import re
from transformers import StoppingCriteria


# Define a stopping condition for generation
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]

        return any(stop_string in current_text for stop_string in self.stop_strings)


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore = [
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None
class LogLossCallback(TrainerCallback):
    def __init__(self, logger,client, current_round):
        self.losses = []
        self.logger = logger
        self.client = client
        self.current_round = current_round
    def on_train_end(self, args, state, control, **kwargs):
        if state.log_history :
            for i in state.log_history:
                if "loss" in i:
                    self.losses.append(i['loss'])
                    self.logger.add_scalar('train_step_acc/client_{}'.format(self.client), i['loss'], self.current_round * state.global_step  + i['step'])


def extract_ground_truth(text):
    return text.split('####')[-1].strip()
def initialize_text_to_text_model(
    model_name: str,
    model_type: str,
    dtype: str,
    tokenizer: str = None,
    flash_attention: bool = False,
):
    assert model_type in ["CausalLM", "ConditionalGeneration"]
    auto_model_class = (
        AutoModelForCausalLM if model_type == "CausalLM" else AutoModelForSeq2SeqLM
    )
    model_config = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    if flash_attention:
        log.info("Using flash attention 2")
        model_config["attn_implementation"] = "flash_attention_2"
    if dtype == "fp32":
        model_config["torch_dtype"] = torch.float32
    elif dtype == "bf16":
        model_config["torch_dtype"] = torch.bfloat16
    elif dtype == "int8":
        quant_8bit_config = dict(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False
        )
        model_config["quantization_config"] = quant_8bit_config
    elif dtype == "nf4":
        quant_4bit_config = dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_config["quantization_config"] = quant_4bit_config
    else:
        raise ValueError("Wrong dtype")

    model = auto_model_class.from_pretrained(**model_config)
    if tokenizer:
        log.info(f"Using custom tokenizer {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def transform_dataset(model_type, tokenizer, dataset, max_length):
    if model_type == "CausalLM":
        dataset.set_transform(lambda x: causalLMEncode(x, tokenizer, max_length))
    elif model_type == "ConditionalGeneration":
        dataset.set_transform(lambda x: SeqToSeqEncode(x, tokenizer, max_length))
    else:
        raise ValueError("Wrong model type")
    return dataset


def train_text_to_text_model(
    run_name: str,
    train_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    **kwargs,
) -> torch.nn.Module:
    # Preprocess the dataset
    train_dataset = preprocess_dataset(train_dataset)

    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // (
        per_device_batch_size * kwargs.get("num_process", 1)
    )
    train_dataset= transform_dataset(
        model_type, tokenizer, train_dataset, max_length
    )

    TrainingArgumentsClass = Seq2SeqTrainingArguments
    if kwargs.get("fed_alg", 'fedavg') == 'fedprox':
        TrainerClass = LogTrainerFedProx
    elif kwargs.get("fed_alg", 'fedavg') == 'scaffold':
        TrainerClass = LogTrainerSCAFFOLD
    else:
        TrainerClass = LogTrainer
    output_dir = f"./results/{run_name}/{kwargs.get('seed')}"
    training_args = TrainingArgumentsClass(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("num_train_epochs", 1),
        max_steps=kwargs.get("max_steps", 200),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_steps=kwargs.get("logging_steps", 1),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
        optim=kwargs.get("optim", "adamw_torch"),
        do_train=True,
        do_eval=False,
        learning_rate=kwargs.get("learning_rate", 5e-5),
        remove_unused_columns=False,  # tokenize the dataset on the fly
        label_names=["labels"],
        seed=kwargs.get("seed", 42),
        ddp_find_unused_parameters=False,
        bf16=True,

        **kwargs.get("training_args", {}),
    )
    """
    eval_accumulation_steps (int, optional) — Number of predictions steps to accumulate the output tensors for,
    before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/NPU/TPU before being moved to the CPU 
    (faster but requires more memory).
    
    if you want to specify compute_metrics for TrainingAguments, you can (should) specify preprocess_logits_for_metrics for Trainer to to avoid
    `cuda out of memory`
    """
    print(kwargs.get("fed_alg", 'fedavg'))
    if kwargs.get("fed_alg", 'fedavg') == 'fedprox':
        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=None,
            callbacks=[kwargs.get("loss_callback", None)],
            prox_mu=kwargs.get("prox_mu", 0.01),
            global_state=kwargs.get("global_state", None)
        )
    elif kwargs.get("fed_alg", 'fedavg') == 'scaffold':
        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=None,
            callbacks=[kwargs.get("loss_callback", None)],
            global_state=kwargs.get("global_state", None),
            local_auxiliary=kwargs.get("auxiliary_model_list", None)[kwargs.get("client", None)],
            global_auxiliary=kwargs.get("global_auxiliary", None),
        )
    else:
        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=None,
            callbacks=[kwargs.get("loss_callback", None)],
        )
    train_result = trainer.train()
    if kwargs.get("fed_alg", 'fedavg') == 'scaffold':
        kwargs.get("auxiliary_model_list", None)[kwargs.get("client", None)], kwargs.get("auxiliary_delta_dict", None)[kwargs.get("client", None)] = trainer.get_auxiliary_param()

    wandb.finish()
    return train_result
def get_auxiliary_dict(fed_alg, num_clients, global_dict):

    if fed_alg =='scaffold':
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [deepcopy(global_auxiliary) for _ in range(num_clients)]    # c_i in SCAFFOLD
        auxiliary_delta_dict = [deepcopy(global_auxiliary) for _ in range(num_clients)]    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*num_clients
        auxiliary_delta_dict = [None]*num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict
def compute_metrics(p: PredictionOutput):
    predictions = p.predictions
    label_ids = p.label_ids # shape (batch_size, seq_len)
    if False:
        # Hard metric: the model must output exactly the same as the target
        # This should be the default evaluation metric for most tasks
        pred = np.argmax(predictions[0], axis=-1)
        num_correct = sum([np.array_equal(pred[i], label_ids[i]) for i in range(len(pred))])
        accuracy = num_correct / len(pred)
    else:
        # Soft metric: we limit the output space to the target space
        # i.e. the model classify the one with higher prob in positive and negative
        # **Use it in cola and mrpc, because it's too hard for vanilla lora**
        # Only suit for the binary classification with each label of 1 token
        label_ids = label_ids[:, 0] # remove the eos token
        unique_labels = np.unique(label_ids)
        flipped_labels = np.ones_like(label_ids) * unique_labels.sum() - label_ids
        predictions = predictions[0][:, 0, :] # remove the eos token # seq_len, tokens
        label_prob = predictions[np.arange(len(predictions)), label_ids]
        flipped_label_prob = predictions[np.arange(len(predictions)), flipped_labels]
        num_correct = sum(label_prob > flipped_label_prob)
        accuracy = num_correct / len(label_prob)

    return {"accuracy": accuracy}

def eval_text_to_text_model(
        run_name: str,
        valid_dataset: Dataset,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        model_type: str,
        per_device_batch_size: int = 1,
        real_batch_size: int = 32,
        max_length: int = None,
        **kwargs,
) -> torch.nn.Module:
    # Preprocess the dataset
    # train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    assert (
            real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // (
            per_device_batch_size * kwargs.get("num_process", 1)
    )
    valid_dataset = transform_dataset(model_type, tokenizer, valid_dataset, max_length)

    eval_steps = (
            int(len(valid_dataset) * kwargs.get("eval_epochs", 1)) // real_batch_size
    )
    TrainingArgumentsClass = Seq2SeqTrainingArguments
    TrainerClass = LogTrainer
    output_dir = f"./results/{run_name}/{kwargs.get('seed')}"
    training_args = TrainingArgumentsClass(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("num_train_epochs", 1),
        max_steps=kwargs.get("max_steps", 200),
        per_device_eval_batch_size=per_device_batch_size,

        evaluation_strategy="steps",
        eval_steps=eval_steps,

        do_train=False,
        do_eval=True,
        remove_unused_columns=False,  # tokenize the dataset on the fly
        eval_accumulation_steps=kwargs.get("eval_accumulation_steps", real_batch_size),
        label_names=["labels"],
        seed=kwargs.get("seed", 42),
        ddp_find_unused_parameters=False,
    )
    """
    eval_accumulation_steps (int, optional) — Number of predictions steps to accumulate the output tensors for,
    before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/NPU/TPU before being moved to the CPU 
    (faster but requires more memory).

    if you want to specify compute_metrics for TrainingAguments, you can (should) specify preprocess_logits_for_metrics for Trainer to to avoid
    `cuda out of memory`
    """
    trainer = TrainerClass(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics if "ConditionalGeneration" in model_type else None,

    )
    eval_result = trainer.evaluate()
    wandb.finish()
    return eval_result


def pissa_init(lora_alpha, lora_rank,init_dict, model):
    param_names = []
    # print(init_dict.keys())
    for key in init_dict.keys():
        if 'lora_A' in key:
            param_names.append(key.replace('.lora_A.weight', ''))
        elif 'lora_B' in key:
            param_names.append(key.replace('.lora_B.weight', ''))

    param_names = set(param_names)
    scale = lora_alpha / lora_rank

    start_time = time.time()
    tracemalloc.start()

    for key in param_names:
        # print(model.state_dict().keys())
        w = model.state_dict()[key + '.base_layer.weight']
        r = lora_rank

        V, S, Uh = torch.linalg.svd(w, full_matrices=False)
        Vr = V[:, : r]
        Sr = S[: r]
        Sr /= scale
        Uhr = Uh[: r]

        B2 = torch.diag(torch.sqrt(Sr)) @ Uhr
        B1 = Vr @ torch.diag(torch.sqrt(Sr))

        init_dict[key + '.lora_B.weight'] = B1
        init_dict[key + '.lora_A.weight'] = B2

        # Res_Vr = V[:, r:]
        # Res_Sr = S[r:]
        # Res_Uh = Uh[r:]
        # model.state_dict()[key+'.weight'].data.copy_(Res_Vr @ torch.diag(Res_Sr) @  Res_Uh)

        temp = model.state_dict()[key + '.base_layer.weight'] - B1 @ B2 * scale
        model.state_dict()[key + '.base_layer.weight'].data.copy_(temp)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1e6} MB")
    print(f"Peak memory usage: {peak / 1e6} MB")

    tracemalloc.stop()

    return init_dict, model

