#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', '1'}:
        return True
    elif value.lower() in {'false', 'no', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--comm_freq', type=float, default=0.5,
                        help="comm freq")
    parser.add_argument('--sub_num_epoch', type=int, default=2,
                        help="comm freq")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='vit', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str_to_bool, default=True,
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")

    parser.add_argument('--gpu', default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_id', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--run_tag', type=str, default='None', help="type \
                        of optimizer")
    parser.add_argument('--zeta_mode', type=str, default='decay',
                        choices=['constant', 'decay'],
                        help="Mode for the Round-Matching coefficient zeta. "
                             "'constant' keeps zeta fixed (recommended for IID); "
                             "'decay' applies a cosine schedule from --zeta down "
                             "to --min_zeta (recommended for Non-IID).")
    parser.add_argument('--zeta', type=float, default=1.0,
                        help="Initial value of the Round-Matching coefficient zeta.")

    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--dir', type=float, default=0,
        help='Dirichlet α for class proportion across clients when --iid 0 (smaller often means '
        'stronger heterogeneity; typical e.g. 0.1, 0.5, 1.0). If 0, get_dataset falls back to α=0.1.'
    )
    parser.add_argument('--lora_alpha', type=int, default=4, help='lora alpha')
    parser.add_argument('--lora_r', type=int, default=2, help='lora r')
    parser.add_argument('--svd_lowrank', type=str_to_bool, default=False,
                        help="svd lowrank")
    parser.add_argument('--svd_iters', type=int, default=32,
                        help="svd iterations")
    parser.add_argument('--sample_size', type=int, default=512,
                        help="sample size")
    parser.add_argument('--min_zeta', type=float, default=0,
                        help="Minimum value of the Round-Matching coefficient "
                             "zeta used by the 'decay' schedule.")
    parser.add_argument('--gamma', type=int, default=64,
                        help="Stabilization hyperparameter gamma for the "
                             "Gradient-Aligned matrix \\hat{W}_{c,ga}^{t}.")
    parser.add_argument('--use_rslora', type=str_to_bool, default=True,
                        help="Whether to use rslora (True/False).")

    # ---- FedSmoothLoRA ablation: "same Round-Matching scaffold, vary only Gradient-Aligned source" ----
    parser.add_argument('--init_source', type=str, default='weight',
                        choices=['weight', 'shared_grad', 'shuffled_grad', 'client_grad'],
                        help="Source used to build the Gradient-Aligned matrix "
                             "\\hat{W}_{c,ga}^{t} under a fixed Round-Matching "
                             "scaffold W_{c,rm}^{t}.")
    parser.add_argument('--align_sample_size', type=int, default=128,
                        help="Sample size for per-client gradient used ONLY for alignment measurement "
                             "when it is not already needed for init construction.")
    parser.add_argument('--log_align', type=str_to_bool, default=True,
                        help="Whether to log per-round Initialization-Gradient Alignment metric.")

    # ---- Observation experiment for "Faster + Client-Specific" analysis ----
    parser.add_argument('--obs_log', type=str_to_bool, default=False,
                        help="Enable the unified observation logging hooks "
                             "(per-step early loss / trajectory alignment / "
                             "inter-client init distance / round acc) used "
                             "to compare FedAvgLoRA / FRLoRA / FedSmoothLoRA.")
    parser.add_argument('--obs_K', type=int, default=10,
                        help="Number of probe SGD steps used by the trajectory "
                             "probe (figures 1.1 and 2.1).")
    parser.add_argument('--obs_method_name', type=str, default='auto',
                        help="Method tag to write in observation CSVs. 'auto' "
                             "lets each script pick its own canonical name.")
    parser.add_argument('--dump_init', type=str_to_bool, default=False,
                        help="When set, ObsLogger additionally dumps per-round "
                             "BA_init (per client) and BA_target (full-data "
                             "gradient SVD-approx) to init_snapshots.pt for the "
                             "init-comparison observation experiment.")

    # ---- Round-to-round global state consistency tracker ----
    parser.add_argument('--track_consistency', type=str_to_bool, default=False,
                        help="If True, log per-round Frobenius norm of the "
                             "global LoRA BA delta and cosine similarity "
                             "between consecutive deltas to "
                             "obs_round_consistency.csv. Used to compare the "
                             "round-to-round trajectory consistency across "
                             "FedAvgLoRA / FRLoRA / FedSmoothLoRA.")

    args = parser.parse_args()
    return args
