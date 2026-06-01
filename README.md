# FedSmoothLoRA

Official implementation of **FedSmoothLoRA: Towards Faster and Smoother Convergence in Federated Low-Rank Adaptation**.

## Overview

FedSmoothLoRA tackles three issues of FedAvg + LoRA: limited update space, inter-round state mismatch, and client-agnostic starting state. At each communication round, the local LoRA is initialized as

$$
\boldsymbol{W}_{c,\mathrm{init}}^{t} = \boldsymbol{\hat{W}}_{c,\mathrm{ga}}^{t} + \zeta \cdot \boldsymbol{W}_{c,\mathrm{rm}}^{t},
$$

- $\boldsymbol{W}_{c,\mathrm{rm}}^{t}$ — **Round-Matching** matrix, aligns the new round with the client's previous local LoRA state.
- $\boldsymbol{\hat{W}}_{c,\mathrm{ga}}^{t}$ — **Gradient-Aligned** matrix, built from the SVD of local full-model gradients on a small calibration mini-batch.
- $\zeta$ — `constant` mode for IID, `decay` (cosine) for Non-IID.

The server performs Full-Rank Aggregation and merges the aggregated LoRA into the backbone, enlarging the effective update space across rounds.

## Setup

```bash
pip install -r requirements.txt
```

## Layout

- `IC_exp/` — image classification (ViT-Small on CIFAR-100).
  - `fedsmoothlora.py` — **FedSmoothLoRA** entry point.
  - `options.py`, `update.py`, `sampling.py`, `utils.py` — argument parser, local update, data partitioning, helpers.
- `NLG_exp/` — natural language generation (LLaMA-3.2-1B on math / code / chat).
  - `fedsmoothlora.py` — **FedSmoothLoRA** entry point.
  - `data.py`, `logTrainer.py`, `merge.py`, `utils.py` — datasets, training callbacks, merge utilities, helpers.

## Quick Start

**Image classification (CIFAR-100, IID).**

```bash
cd IC_exp
python fedsmoothlora.py \
  --dataset cifar100 --iid 1 \
  --lora_r 2 --lora_alpha 4 --use_rslora True --gamma 256 \
  --zeta_mode constant --zeta 1.0 \
  --run_tag cifar100_iid_seed0
```

**Natural language generation (Code-Feedback, IID).**

```bash
cd NLG_exp
python fedsmoothlora.py \
  --model_id meta-llama/Llama-3.2-1B --dataset_name codefeedback \
  --num_rounds 10 --local_steps 200 --num_clients 3 \
  --real_batch_size 32 --per_device_batch_size 4 --max_length 1024 \
  --learning_rate 2e-5 --lora_rank 32 --lora_alpha 64 \
  --scale stable --stable_gamma 64 --svd_iters 8 \
  --zeta_mode constant --zeta 1.0 \
  --ga_bsz 4 --sample_size 32 --device cuda --run_tag seed_0
```

For Non-IID client distributions, switch to `--non_iid True --zeta_mode decay`.

## Key Hyperparameters

| Argument | Symbol | Notes |
| --- | --- | --- |
| `lora_r` / `lora_rank` | $r$ | LoRA rank. |
| `lora_alpha` | $\alpha$ | LoRA scaling; with `--use_rslora` / `--scale stable` use $s=\alpha/\sqrt{r}$. |
| `gamma` / `stable_gamma` | $\gamma$ | Stabilization for $\boldsymbol{\hat{W}}_{c,\mathrm{ga}}^{t}$. |
| `ga_bsz` / `sample_size` | — | Calibration mini-batch size for the local gradient estimate. |
| `zeta_mode` | $\zeta_{\mathrm{mode}}$ | `constant` (IID) or `decay` (Non-IID, cosine). |
| `zeta`, `min_zeta` | $\zeta$ | Initial / minimum value of the Round-Matching coefficient. |
| `svd_iters` | — | Iterations for randomized low-rank SVD in `SVDApprox`. |
