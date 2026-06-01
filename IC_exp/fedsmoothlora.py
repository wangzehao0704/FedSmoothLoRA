#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
"""FedSmoothLoRA on image classification.

Implements FedSmoothLoRA (Algorithm 2 in the paper). At each communication
round, the client-side LoRA is initialized as

    W_{c,init}^{t} = \hat{W}_{c,ga}^{t} + zeta * W_{c,rm}^{t},

where W_{c,rm}^{t} is the Round-Matching matrix that aligns the new round
with the client's previous local LoRA state, and \hat{W}_{c,ga}^{t} is the
Gradient-Aligned matrix built from a layer-wise estimate of the local
full-model gradient. ``zeta`` is the Round-Matching coefficient, supporting
the ``constant`` mode (zeta=1, used for IID) and the ``decay`` mode (cosine
schedule, used for Non-IID).
"""
import math
import os
import copy
import time
import numpy as np
import timm
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from utils import (
    DATASET_NUM_MAP,
    average_weights,
    cosine_learning_rate,
    cosine_zeta_schedule,
    exp_details,
    get_dataset,
    matrix_addition_every_layer,
    matrix_multiply_every_layer,
    matrix_subtraction_every_layer,
    matrix_truncated_svd_every_layer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    start_time = time.time()
    args = args_parser()
    args.num_classes = DATASET_NUM_MAP[args.dataset]

    path_project = os.path.abspath('../..')
    timestamp = str(
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
    )
    exp_name = "FedSmoothLoRA-" + args.run_tag + timestamp
    logger = SummaryWriter(os.path.join("./logs/", exp_name))
    exp_details(args)
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    train_dataset, test_dataset, user_groups = get_dataset(args)
    count_list = [len(v) for _, v in user_groups.items()]

    global_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=args.num_classes)
    print(global_model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_rslora=args.use_rslora,
        lora_dropout=0.05,
        bias="none",
        target_modules=['qkv', 'head'],
    )
    global_model = get_peft_model(global_model, peft_config=peft_config)
    global_model.print_trainable_parameters()

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = copy.deepcopy(get_peft_model_state_dict(global_model))
    local_weights_list = [copy.deepcopy(global_weights) for _ in range(args.num_users)]

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    client_step = [0] * args.num_users

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        local_losses = []
        print(f'\n | Global Training Round : {epoch} |\n')
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        lr = cosine_learning_rate(epoch, args.epochs, initial_lr=args.lr)
        model_dict = copy.deepcopy(global_model.state_dict())
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, lr=lr, idx=idx,
                                      idxs=user_groups[idx], logger=logger)
            if epoch == 0:
                # First round: only the Gradient-Aligned matrix is used
                # (W_{c,rm}^{0} = 0 by definition).
                local_named_grads = local_model.estimate_gradient(
                    model=copy.deepcopy(global_model),
                    sample_size=args.sample_size,
                    bsz=args.local_bs,
                )
                ga_init = local_model.get_ga_dict(local_named_grads, args.gamma)
                ga_ba = matrix_multiply_every_layer(ga_init, 'cuda')
                # Pre-shift the backbone by -s * \hat{W}_{c,ga}^{t} so that
                # the effective merged weight is W_c^{t} - s \hat{W}_{c,ga}^{t}
                # as described in Algorithm 2.
                for name, param in global_model.named_parameters():
                    if 'base_layer.weight' in name:
                        if args.use_rslora:
                            param.data = param.data - (args.lora_alpha / math.sqrt(args.lora_r)) * ga_ba[
                                name.replace('base_layer.weight', 'lora_BA.weight')].to(device)
                        else:
                            param.data = param.data - (args.lora_alpha / args.lora_r) * ga_ba[
                                name.replace('base_layer.weight', 'lora_BA.weight')].to(device)
                set_peft_model_state_dict(global_model, ga_init)

            else:
                # Build the Round-Matching matrix
                #   W_{c,rm}^{t} = B_c^{t-1} A_c^{t-1} - B_s^{t} A_s^{t},
                # weighted by zeta (constant or cosine-decayed).
                global_ba = matrix_multiply_every_layer(global_weights_svd, 'cuda')
                local_ba = matrix_multiply_every_layer(local_weights_list[idx], 'cuda')

                if args.zeta_mode == 'constant':
                    rm_ba = matrix_subtraction_every_layer(local_ba, global_ba, 'cuda', args.zeta)
                elif args.zeta_mode == 'decay':
                    current_zeta = cosine_zeta_schedule(
                        current_round=epoch,
                        total_rounds=args.epochs,
                        initial_zeta=args.zeta,
                        min_zeta=args.min_zeta,
                    )
                    rm_ba = matrix_subtraction_every_layer(local_ba, global_ba, 'cuda', current_zeta)
                else:
                    raise NotImplementedError(f"Unknown zeta_mode: {args.zeta_mode}")

                # Build the Gradient-Aligned matrix \hat{W}_{c,ga}^{t} from a
                # layer-wise estimate of the local full-model gradient. The
                # gradient is estimated on the backbone shifted by the current
                # zeta-scaled Round-Matching matrix so that the gradient
                # signal corresponds to the actual local starting point.
                local_named_grads = local_model.estimate_gradient(
                    model=copy.deepcopy(global_model),
                    sample_size=args.sample_size,
                    bsz=args.local_bs,
                    add_param=rm_ba,
                )
                ga_init = local_model.get_ga_dict(local_named_grads, args.gamma)

                ga_ba = matrix_multiply_every_layer(ga_init, 'cuda')
                # Local LoRA initialization:
                #   W_{c,init}^{t} = \hat{W}_{c,ga}^{t} + zeta * W_{c,rm}^{t}
                init = matrix_truncated_svd_every_layer(
                    matrix_addition_every_layer(ga_ba, rm_ba, 'cuda'),
                    args.lora_r, 8, False, 'cuda',
                )
                # Merge -s * \hat{W}_{c,ga}^{t} into the backbone so that local
                # training proceeds with the shifted backbone.
                for name, param in global_model.named_parameters():
                    if 'base_layer.weight' in name:
                        if args.use_rslora:
                            param.data = param.data - (args.lora_alpha / math.sqrt(args.lora_r)) * ga_ba[
                                name.replace('base_layer.weight', 'lora_BA.weight')].to(device)
                        else:
                            param.data = param.data - (args.lora_alpha / args.lora_r) * ga_ba[
                                name.replace('base_layer.weight', 'lora_BA.weight')].to(device)
                set_peft_model_state_dict(global_model, init)
            w, loss, client_step = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, client_step=client_step)
            # Effective uploaded LoRA update:
            #   B_c^t A_c^t = SVDApprox(B_c^t A_c^t - \hat{W}_{c,ga}^t; r)
            temp_ba = matrix_multiply_every_layer(copy.deepcopy(w), 'cuda')
            ga_ba = matrix_multiply_every_layer(ga_init, 'cuda')
            local_weights_list[idx] = matrix_truncated_svd_every_layer(
                matrix_subtraction_every_layer(temp_ba, ga_ba, 'cuda'),
                args.lora_r, args.svd_iters, args.svd_lowrank, 'cuda',
            )
            local_losses.append(copy.deepcopy(loss))
            global_model.load_state_dict(model_dict)
            global_model.eval()
            set_peft_model_state_dict(global_model, copy.deepcopy(local_weights_list[idx]))
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print('Client_{}: {:.2f}%'.format(idx, 100 * test_acc))
            logger.add_scalar('test_loss/client_{}'.format(idx), test_loss, epoch)
            logger.add_scalar('test_acc/client_{}'.format(idx), test_acc, epoch)
            global_model.train()
            del local_model
        del model_dict

        # Server-side Full-Rank Aggregation followed by rank-r SVD projection.
        local_ba_weights = matrix_multiply_every_layer(local_weights_list, 'cuda')
        global_weights = average_weights(local_ba_weights, count_list, idxs_users)
        global_weights_svd = matrix_truncated_svd_every_layer(
            global_weights, args.lora_r, args.svd_iters, args.svd_lowrank, 'cuda',
        )

        set_peft_model_state_dict(global_model, global_weights_svd)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        global_model.eval()

        list_acc, list_loss = [], []
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, lr=lr, idx=idx,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            del local_model
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        logger.add_scalar('test_acc/global', test_acc, epoch)
        logger.add_scalar('test_loss/global', test_loss, epoch)
        logger.add_scalar('train_acc/global', sum(list_acc) / len(list_acc), epoch)
        logger.add_scalar('train_loss/global', sum(list_loss) / len(list_loss), epoch)
        print(f' \nAvg Training Stats after {epoch} global rounds:')
        print(f'|---- Avg Training Loss : {np.mean(np.array(train_loss))}')
        print(f"|---- Test Loss: {test_loss}")
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        # Merge the aggregated server-side LoRA into the backbone, enlarging
        # the effective parameter update space across rounds.
        global_model = global_model.merge_and_unload()
        global_model = get_peft_model(global_model, peft_config=peft_config)
    global_model = global_model.unload()
    torch.save(global_model.state_dict(), os.path.join("./ckpts/", exp_name))
