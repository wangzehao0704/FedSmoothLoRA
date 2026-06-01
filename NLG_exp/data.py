from datasets import load_dataset, Dataset, concatenate_datasets
import typing as tp
import functools
import os
import pickle
import logging
import hashlib
import json
log = logging.getLogger(__name__)


def cache_to_disk(root_datadir="data_cache"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_filename = root_datadir + "/" + f"{func_name}.pkl"
            args_str = "_".join(map(str, args))
            kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())
            params_str = f"{args_str}_{kwargs_str}"
            params_hash = hashlib.md5(params_str.encode()).hexdigest()

            cache_filename = os.path.join(root_datadir, f"{func_name}_{params_hash}.pkl")
            print("cache_filename =", cache_filename)

            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    print(f"Loading cached data for {func.__name__} {params_str}")
                    return pickle.load(f)

            result = func(*args, **kwargs)

            print("caching " + cache_filename)
            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
                print(f"Cached data for {func.__name__}")

            hash_table_filename = os.path.join(root_datadir, "hash_table.txt")
            if not os.path.exists(hash_table_filename):
                with open(hash_table_filename, "w"):
                    pass
            with open(hash_table_filename, "a") as f:
                f.write(f"{cache_filename}: {params_str}\n")

            return result

        return wrapper

    return decorator



template_with_input = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

template_wo_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def load_dolly(model_id=None, seed=42, max_tokens=512):
    dataset = load_dataset("/path/to/yourdata/")

    def alpaca_preprocess(instruction, input, output, cate):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output, "cate": cate}

    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["context"], e["response"], cate=e["category"]),
    )
    # we sample 10% of the training set as validation set
    train_set = dataset["train"].train_test_split(test_size=0.1)["train"]
    return train_set,

def load_meta_math(model_id=None, seed=42, max_tokens=512):
    dataset = load_dataset("/path/to/yourdata/", split="train")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=seed)
    from tqdm import tqdm

    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if (
            len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens
            or "GSM" not in sample["type"]
        ):
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif count >= 100000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    return train_set

def load_codefeedback(model_id=None, seed=42, max_tokens=1024):
    dataset = load_dataset("/path/to/yourdata/", split="train")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess(data):
        y = data["answer"]
        y = "```".join(y.split("```")[:2]) + "```"  # only keep the first code block
        return {
            "x": template_wo_input.format(instruction=data["query"]),
            "y": y,
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=seed)
    from tqdm import tqdm

    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample["answer"]:
            continue
        if len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:
            train_samples.append(processed_sample)
        elif count >= 100000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    return train_set

DATASET_MAP = {
    "meta_math": load_meta_math,
    "codefeedback": load_codefeedback,
    "dolly": load_dolly,

}