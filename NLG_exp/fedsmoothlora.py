"""FedSmoothLoRA on natural language generation tasks.

Implements FedSmoothLoRA (Algorithm 2 in the paper) for federated fine-tuning
of large language models. At each communication round, the client-side LoRA
is initialized as

    W_{c,init}^{t} = \\hat{W}_{c,ga}^{t} + zeta * W_{c,rm}^{t},

where the Round-Matching matrix W_{c,rm}^{t} preserves cross-round
optimization continuity and the Gradient-Aligned matrix \\hat{W}_{c,ga}^{t}
incorporates client-specific optimization signals from a layer-wise estimate
of the local full-model gradient. The coefficient ``zeta`` supports a
``constant`` mode (zeta=1, used for IID) and a ``decay`` mode (cosine
schedule, used for Non-IID).
"""
from copy import deepcopy

from fire import Fire
import time

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from peft import (
    LoraConfig,
    LoraGAConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils.lora_ga_utils import (
    LoraGAContext,
    estimate_gradient,
    save_loraga_model_final,
    save_loraga_model_init,
)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from accelerate import Accelerator

from utils import *
from data import DATASET_MAP
import wandb
import os

from tensorboardX import SummaryWriter


def main(lora_alpha=16, lora_rank=8, sample_size=128, seed=31, num_rounds=10, num_clients=3, local_steps=100,
         learning_rate=2e-5, real_batch_size=128, per_device_batch_size=2, dataset_name="meta_math", svd_iters=8,
         max_length=1024, stable_gamma=64, run_tag='None', svd_lowrank=True, scale='stable',
         model_id="ahxt/LiteLlama-460M-1T", model_type="CausalLM", device='cpu', model_dtype='fp32',
         zeta_mode='constant', zeta=1.0, ga_bsz=2, use_rslora=False, min_zeta=0.5, non_iid=False, alpha=1.0,
         fed_alg='fedavg', prox_mu=0.01, fedopt_eta=1e-3, fedopt_tau=1e-3, fedopt_beta1=0.9, fedopt_beta2=0.99):
    """Run FedSmoothLoRA on a natural language generation task.

    Args:
        zeta_mode: Mode for the Round-Matching coefficient zeta. ``constant``
            keeps zeta fixed at ``zeta`` (recommended for IID); ``decay``
            applies a cosine schedule from ``zeta`` down to ``min_zeta``
            (recommended for Non-IID).
        zeta: Initial value of the Round-Matching coefficient.
        min_zeta: Minimum value of zeta for the ``decay`` mode.
        stable_gamma: Stabilization hyperparameter gamma for the
            Gradient-Aligned matrix \\hat{W}_{c,ga}^{t} (Eq. (8) in the
            paper).
        ga_bsz: Calibration mini-batch size used to estimate the local
            full-model gradient for the Gradient-Aligned matrix.
    """
    params = locals()
    for param, value in params.items():
        print(f"{param}: {value}")
    accelerator = Accelerator()
    print("Accelerator num processes:", accelerator.num_processes, )
    config = dict(
        model="llama",
        d=dataset_name,
        a=lora_alpha,
        r=lora_rank,
        s=sample_size,
        sd=seed,
    )
    timestamp = str(
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
    )
    exp_name = "FedSmoothLoRA-" + run_tag + timestamp
    logger = SummaryWriter(os.path.join("./logs/", exp_name))
    dataset_func = DATASET_MAP[dataset_name]

    total_train_set = dataset_func(model_id, seed, max_length)
    print("Training set size:", len(total_train_set))
    local_train_set = split_dataset(num_clients, seed, total_train_set, dataset_name, alpha, non_iid)
    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if accelerator.is_local_main_process:
        print(model)
    peft_config = LoraGAConfig(
        bsz=ga_bsz,
        target_modules=find_all_linear_modules(model=model),
        lora_alpha=config["a"],
        r=config["r"],
        iters=config["s"] // ga_bsz,
        scale=scale,
        use_rslora=use_rslora,
        stable_gamma=stable_gamma,
    )
    temp_peft_config = LoraConfig(
        target_modules=find_all_linear_modules(model=model),
        lora_alpha=config["a"],
        r=config["r"],
        use_rslora=use_rslora,
    )
    save_dir = os.path.join("./snapshot", exp_name)

    model = get_peft_model(model, temp_peft_config)
    # ===== Define the global and local models =====
    global_dict = deepcopy(get_peft_model_state_dict(model))
    global_dict_ba = matrix_multiply_every_layer(global_dict, device)
    local_dict_list = [deepcopy(global_dict) for _ in range(num_clients)]
    sample_num_list = [len(local_train_set[i]) for i in range(num_clients)]
    model = model.unload()
    for current_round in tqdm(range(num_rounds)):

        clients_this_round = get_clients_this_round(num_clients, current_round)

        print(f">> ==================== Round {current_round + 1} : {clients_this_round} ====================")

        model_state_dict = deepcopy(model.to('cpu').state_dict())
        for client in range(num_clients):
            loss_callback = LogLossCallback(logger, client, current_round)
            new_lr = cosine_learning_rate(current_round, num_rounds, learning_rate,
                                          1e-6)  # manually schedule the learning rate

            train_set = get_dataset_this_round(local_train_set[client], current_round, real_batch_size,
                                               local_steps)  # get the required sub-dataset for this round

            model.requires_grad_()
            model.to('cpu')
            if current_round >= 1:
                # Build the Round-Matching matrix
                #   W_{c,rm}^{t} = B_c^{t-1} A_c^{t-1} - B_s^{t} A_s^{t},
                # weighted by zeta (constant or cosine-decayed).
                local_dict_ba = matrix_multiply_every_layer(local_dict_list[client], device)
                global_dict_ba = matrix_multiply_every_layer(global_dict, device)
                if zeta_mode == 'decay':
                    current_zeta = cosine_zeta_schedule(
                        current_round=current_round,
                        total_rounds=num_rounds,
                        initial_zeta=zeta,
                        min_zeta=min_zeta,
                    )
                    rm_dict_ba = matrix_subtraction_every_layer(local_dict_ba, global_dict_ba, device, current_zeta)
                elif zeta_mode == 'constant':
                    rm_dict_ba = matrix_subtraction_every_layer(local_dict_ba, global_dict_ba, device, zeta)
                else:
                    raise NotImplementedError(f"Unknown zeta_mode: {zeta_mode}")
                del local_dict_ba
                del global_dict_ba
                # Pre-shift the backbone by + s * W_{c,rm}^{t} so that the
                # Gradient-Aligned matrix is estimated on the actual local
                # starting point used for local training.
                for name, param in model.named_parameters():
                    if name.replace('weight', 'lora_BA.weight').replace('model',
                                                                        'base_model.model.model') in rm_dict_ba:
                        if use_rslora:
                            param.data = param.data + (peft_config.lora_alpha / math.sqrt(peft_config.r)) * rm_dict_ba[
                                name.replace('weight', 'lora_BA.weight').replace('model', 'base_model.model.model')].to(
                                'cpu')
                        else:
                            param.data = param.data + (peft_config.lora_alpha / peft_config.r) * rm_dict_ba[
                                name.replace('weight', 'lora_BA.weight').replace('model', 'base_model.model.model')].to(
                                'cpu')

                # Build the Gradient-Aligned matrix \hat{W}_{c,ga}^{t} from a
                # layer-wise estimate of the local full-model gradient on a
                # small calibration mini-batch.
                if isinstance(train_set, list):
                    temp_set = train_set[: peft_config.bsz * peft_config.iters]
                else:
                    temp_set = train_set.select(range(peft_config.bsz * peft_config.iters))
                transform_dataset(
                    model_type=model_type,
                    dataset=temp_set,
                    tokenizer=tokenizer,
                    max_length=peft_config.max_length,
                )
                dataloader = torch.utils.data.DataLoader(temp_set, batch_size=peft_config.bsz)

                named_grad = estimate_gradient(
                    model=model,
                    dataloader=dataloader,
                    accelerator=accelerator,
                    quant_flag=False,
                )
                model.load_state_dict(model_state_dict)
                with LoraGAContext(model=model, named_grad=named_grad):
                    model = get_peft_model(model=model, peft_config=peft_config)
                ga_dict = deepcopy(get_peft_model_state_dict(model))
                ga_dict_ba = matrix_multiply_every_layer(ga_dict, device)

                # Local LoRA initialization:
                #   W_{c,init}^{t} = \hat{W}_{c,ga}^{t} + zeta * W_{c,rm}^{t}
                init_dict_ba = matrix_addition_every_layer(ga_dict_ba, rm_dict_ba, device)
                del ga_dict_ba
                del rm_dict_ba
                init_dict = matrix_truncated_svd_every_layer(init_dict_ba, peft_config.r, svd_iters, svd_lowrank, device)
                set_peft_model_state_dict(model, init_dict)
                del init_dict

            else:
                # First round: only the Gradient-Aligned matrix is used
                # (W_{c,rm}^{0} = 0 by definition).
                if isinstance(train_set, list):
                    temp_set = train_set[: peft_config.bsz * peft_config.iters]
                else:
                    temp_set = train_set.select(range(peft_config.bsz * peft_config.iters))
                transform_dataset(
                    model_type=model_type,
                    dataset=temp_set,
                    tokenizer=tokenizer,
                    max_length=peft_config.max_length,
                )
                dataloader = torch.utils.data.DataLoader(temp_set, batch_size=peft_config.bsz)

                named_grad = estimate_gradient(
                    model=model,
                    dataloader=dataloader,
                    accelerator=accelerator,
                    quant_flag=False,
                )

                with LoraGAContext(model=model, named_grad=named_grad):
                    model = get_peft_model(model=model, peft_config=peft_config)
                ga_dict = deepcopy(get_peft_model_state_dict(model))

            print("finish get_peft_model=================================================")
            train_results = train_text_to_text_model(
                run_name=os.path.join("peft_test", exp_name),
                train_dataset=train_set,
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                max_steps=local_steps,
                per_device_batch_size=per_device_batch_size,
                real_batch_size=real_batch_size * accelerator.num_processes,
                max_length=max_length,
                use_loraplus=False,
                loraplus_lr_ratio=None,
                bf16=(model_dtype == "bf16"),
                learning_rate=new_lr,
                num_process=accelerator.num_processes,
                gradient_checkpointing=True,
                seed=seed,
                do_eval=False,
                training_args=dict(
                    lr_scheduler_type="constant",
                    max_grad_norm=1.0,
                    warmup_ratio=0.03,
                    weight_decay=0.0,
                ),
                loss_callback=loss_callback,
            )
            # Effective uploaded LoRA update:
            #   B_c^t A_c^t = SVDApprox(B_c^t A_c^t - \hat{W}_{c,ga}^t; r)
            temp_dict = deepcopy(get_peft_model_state_dict(model))  # copy is needed!
            temp_dict_ba = matrix_multiply_every_layer(temp_dict, device)

            ga_dict_ba = matrix_multiply_every_layer(ga_dict, device)
            diff_dict_ba = matrix_subtraction_every_layer(temp_dict_ba, ga_dict_ba, device)
            del temp_dict_ba
            del ga_dict_ba
            del temp_dict
            local_dict_list[client] = deepcopy(
                matrix_truncated_svd_every_layer(diff_dict_ba, peft_config.r, svd_iters, svd_lowrank,
                                                 device))  # copy is needed!
            del diff_dict_ba
            model = model.unload()
            model.load_state_dict(model_state_dict)
            print(f"Training Loss for round {current_round} of {client}: {train_results.training_loss}")

            logger.add_scalar('train_acc_round/client_{}'.format(client), train_results.training_loss, current_round)
            # ===== Server aggregates the local models =====
        del model_state_dict
        # Server-side Full-Rank Aggregation followed by rank-r SVD projection.
        global_dict_ba = matrix_multiply_every_layer(global_dict, device)

        global_dict_ba = global_aggregate(global_dict_ba, matrix_multiply_every_layer(local_dict_list, device),
                                          sample_num_list, clients_this_round)
        global_dict = matrix_truncated_svd_every_layer(global_dict_ba, peft_config.r, svd_iters, svd_lowrank,
                                                       device)  # Update global model
        del global_dict_ba
        model = get_peft_model(model, temp_peft_config)
        set_peft_model_state_dict(model, global_dict)  # Update global model
        # Merge the aggregated server-side LoRA into the backbone, enlarging
        # the effective parameter update space across rounds.
        model = model.merge_and_unload()
        if current_round == num_rounds - 1:
            model.save_pretrained(os.path.join(save_dir, f"full-{current_round + 1}"))
            tokenizer.save_pretrained(os.path.join(save_dir, f"full-{current_round + 1}"))


if __name__ == "__main__":
    Fire(main)
