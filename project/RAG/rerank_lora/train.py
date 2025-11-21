import math
import json
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer



class Mydataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def format_input(context, query):
    def _format_query(query: str) -> str:
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return prefix + "<Instruct>: {instruction}\n<Query>: {query}\n".format(instruction=instruction, query=query)

    def _format_doc(doc: str) -> str:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return "<Document>: " + doc + suffix
    
    return _format_query(query) + _format_doc(context)


def prepare_dataloader(tokenizer, train_data_path, eval_data_path, batch_size):
    train_dataset = Mydataset(train_data_path)
    eval_dataset = Mydataset(eval_data_path)

    def collate_fn(batch):
        queries = [item["query"] for item in batch]
        contexts = [item["context"] for item in batch]
        inputs = [format_input(context, query) for context, query in zip(contexts, queries)]
        encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=1024)
        scores = torch.tensor([item["label"] for item in batch])

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "scores": scores
        }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, eval_dataloader


def evaluate(
    model,
    eval_dataloader,
    token_yes_id,
    accelerator : Accelerator,
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            pooling_logits = logits[:, -1, :]
            batch_scores = torch.nn.functional.sigmoid(pooling_logits, dim=1)[:, token_yes_id]
            scores = batch["scores"].to(batch_scores.dtype)
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(batch_scores, scores)
            loss = accelerator.gather_for_metrics(loss)
            total_loss += loss.mean().item()
    return total_loss / len(eval_dataloader)


def train(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
    token_yes_id,
    accelerator : Accelerator,
    num_epochs,
    log_steps,
    save_steps,
    resume=None
):
    global_step = 0
    resume_epoch = 0
    resume_step = 0
    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
        resume_step = global_step = accelerator.step
        resume_epoch = resume_step // steps_per_epoch
        resume_step = resume_step % steps_per_epoch
    for ep in range(resume_epoch, num_epochs):
        model.train()
        activedataloader = train_dataloader
        if resume and ep == resume_epoch:
            activedataloader = accelerator.skip_first_batches(activedataloader, resume_step * accelerator.gradient_accumulation_steps)
        bar = tqdm(activedataloader, total=len(activedataloader), desc=f"Training Epoch {ep}")
        for batch in bar:
            outputs = model(**batch)
            logits = outputs.logits
            pooling_logits = logits[:, -1, :]
            batch_scores = torch.nn.functional.sigmoid(pooling_logits)[:, token_yes_id]
            scores = batch["scores"].to(batch_scores.dtype)
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(batch_scores, scores)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % log_steps == 0:
                    loss = accelerator.reduce(loss, reduction="mean")
                    if accelerator.is_main_process:
                        bar.write(f"Epoch {ep} Step {global_step}, train loss: {loss.item()}")
                    accelerator.log({"train/loss": loss.item()}, step=global_step)
                if global_step % save_steps == 0:
                    if accelerator.is_main_process:
                        bar.write(f"Saving model at step {global_step}...")
                    accelerator.save_state(accelerator.project_dir + f"/checkpoints/step_{global_step}")
                    accelerator.unwrap_model(model).save_pretrained(
                        save_directory=accelerator.project_dir + f"/checkpoints/step_{global_step}/model",
                        is_main_process=accelerator.is_main_process,
                    ) # type: ignore
        loss = evaluate(model, eval_dataloader, token_yes_id, accelerator)
        accelerator.print(f"Epoch {ep} Step {global_step}, eval loss: {loss}")
        accelerator.log({"eval/loss": loss}, step=global_step)
    accelerator.end_training()


def main():
    checkpoint = "models"
    train_data_path = "rerank/data/train.json"
    eval_data_path = "rerank/data/eval.json"
    checkpoint = "models/Qwen/Qwen3-Reranker-4B"
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir="./rerank"
    )
    accelerator.init_trackers("log")

    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, padding_side="left")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    train_dataloader, eval_dataloader = prepare_dataloader(tokenizer, train_data_path, eval_data_path, batch_size=2)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    token_yes_id = tokenizer.convert_tokens_to_ids("yes")
    train(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        token_yes_id,
        accelerator,
        resume=None,
        num_epochs=3,
        log_steps=20,
        save_steps=20
    )
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()




