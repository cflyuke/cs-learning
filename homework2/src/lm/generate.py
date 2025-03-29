import argparse
import json
import os
import math

import tiktoken
import torch
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    logits = logits / temperature
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes; computes perplexity

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    tokenized_prefixes = [tokenizer.encode(prefix) for prefix in prefixes]
    max_len = max(len(tokens) for tokens in tokenized_prefixes)
    input_ids = torch.full(
        (len(prefixes), max_len),
        tokenizer.eot_token,
        dtype=torch.long,
        device=device
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    
    for i, tokens in enumerate(tokenized_prefixes):
        input_ids[i, -len(tokens):] = torch.tensor(tokens, device=device)
        attention_mask[i, -len(tokens):] = 1.0
    
    generations = []
    perplexity = 0.0
    total_log_prob = 0.0
    total_tokens = 0.0

    for _ in trange(max_new_tokens):
        logits = model(input_ids, attention_mask=attention_mask)

        probs = softmax_with_temperature(logits[:, -1, :], temperature=temperature)
        log_probs = torch.log(probs)
        token_ids = torch.multinomial(probs, num_samples=1)
        for i in range(len(prefixes)):
            total_log_prob += log_probs[i, token_ids[i]].item()
            total_tokens += 1
        input_ids = torch.cat([input_ids, token_ids], dim = 1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(token_ids, dtype=torch.float(32))], dim = 1)

    for i in range(len(prefixes)):
        generated_ids = input_ids[i, max_len:].tolist()
        generations.append(tokenizer.decode(generated_ids))
    
    perplexity = math.exp(-total_log_prob/total_tokens)

    print(f"Perplexity: {perplexity}")
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
