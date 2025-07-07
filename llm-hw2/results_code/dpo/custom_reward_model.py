#!/usr/bin/env python3
"""
自定义奖励模型加载器
用于加载align-anything训练的奖励模型，无需依赖align-anything库
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, Qwen2Model, Qwen2PreTrainedModel
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput


@dataclass
class ScoreModelOutput(ModelOutput):
    """奖励模型输出"""
    scores: torch.FloatTensor | None = None  # size = (B, L, D)
    end_scores: torch.FloatTensor | None = None  # size = (B, D)
    last_hidden_state: torch.FloatTensor | None = None  # size = (B, L, E)
    end_last_hidden_state: torch.FloatTensor | None = None  # size = (B, E)
    end_index: torch.LongTensor | None = None  # size = (B,)


class CustomQwen2RewardModel(Qwen2PreTrainedModel):
    """自定义Qwen2奖励模型，兼容align-anything训练的模型"""

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)
        
        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> ScoreModelOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_state = outputs.hidden_states[-1]
        scores = self.score_head(last_hidden_state).float()
        B, L, _ = last_hidden_state.size()

        if attention_mask is None:
            if B > 1:
                raise ValueError("'attention_mask' is required when batch size > 1.")
            attention_mask = last_hidden_state.new_ones(B, L, dtype=torch.bool)

        # 找到每个序列的最后一个有效token位置
        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])  # size = (B,)
        
        # 获取最后一个有效位置的hidden state和score
        end_last_hidden_state = torch.gather(
            last_hidden_state,
            dim=1,
            index=(
                end_index.to(last_hidden_state.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, last_hidden_state.size(-1))
            ),
        )
        end_scores = torch.gather(
            scores,
            dim=1,
            index=(
                end_index.to(scores.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, scores.size(-1))
            ),
        )
        
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)  # size = (B, E)
        end_scores = end_scores.squeeze(dim=1)  # size = (B, D)

        return ScoreModelOutput(
            scores=scores,
            end_scores=end_scores,
            last_hidden_state=last_hidden_state,
            end_last_hidden_state=end_last_hidden_state,
            end_index=end_index,
        )


def load_custom_reward_model(model_path, device="cpu", torch_dtype=torch.float32):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = CustomQwen2RewardModel(config)
    
    state_dict = torch.load(
        f"{model_path}/pytorch_model.bin", 
        map_location=device,
        weights_only=True
    )
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.to(torch_dtype)
    model.eval()
    
    return model, tokenizer


def calculate_reward_score_with_custom_model(model, tokenizer, question, response, device="cpu"):
    """
    使用自定义奖励模型计算奖励分数
    """
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        # 检查模型类型
        if isinstance(model, CustomQwen2RewardModel):
            # 使用自定义奖励模型
            outputs = model(**inputs)
            reward_score = outputs.end_scores.squeeze().item()
        else:
            # 使用标准模型的fallback方法
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']

            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            
            last_token_hidden = last_hidden_state[
                torch.arange(batch_size), sequence_lengths
            ]
            
            reward_score = last_token_hidden.mean().item()
    
    return reward_score
