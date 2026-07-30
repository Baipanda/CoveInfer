"""
    by bjy
    1. class LlamaEmbedding 人为“剥离 Embedding 层”
    2. LlamaLossFunc：Embedding 对齐损失
    3. 三种噪声函数：标准 DP-forward 工具箱：Gas_noise、Lap_noise、Exp_noise
    4. add_*_noise_*：核心用途 —— embedding DP 消融实验
    5. normalize_and_add_noise：补救 embedding 崩坏
"""
import json
import logging
import pdb
import random

import torch.nn.functional as F
from torch import nn
import torch
from torch.nn.functional import pdist
import numpy as np


def get_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    # 向文件输出
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 向终端输出
    console = logging.StreamHandler()
    console.setLevel(level=logging.DEBUG)

    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


class lConfig:
    load_in_8bit = False
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05  # "down_proj", "o_proj","embed_tokens"
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    bias = "none"
    task_type = "CAUSAL_LM"


class LlamaEmbeddingConfig:
    pad_token_id = 0
    vocab_size = 32000
    hidden_size = 4096


class DpConfig:
    noise_factor = 2.60
    lap_epsilon = 4.0
    # Llama2 1.8
    # 0 表示高斯噪声， 1表示拉普拉斯噪声
    noise_type = 1
    sensitivity = 1.0
    emb_add_noise = True


class LlamaEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding_idx = LlamaEmbeddingConfig.pad_token_id
        self.vocab_size = LlamaEmbeddingConfig.vocab_size
        self.hidden_size = LlamaEmbeddingConfig.hidden_size
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class LlamaLossFunc(nn.Module):
    def __init__(self):
        super(LlamaLossFunc, self).__init__()

    def forward(self, pred_y, true_y, embed_pred, embed_true):
        euc_dis = torch.norm(embed_true - embed_pred, dim=1)
        loss1 = torch.mean(euc_dis)

        loss2 = F.cross_entropy(
            pred_y.view(-1, LlamaEmbeddingConfig.vocab_size),
            true_y.view(-1).to(pred_y.device),
        )
        loss = loss2 + loss1
        return loss


def Gas_noise(score, mean=0, std=1):
    # 生成高斯噪声
    noise_factor = DpConfig.noise_factor
    noise = noise_factor * np.random.normal(loc=mean, scale=std, size=1)
    # pdb.set_trace()
    return score + noise


# 生成满足 ε-差分隐私的噪声矩阵。
def Lap_noise(score, epsilon, sensitivity):
    noise_b = sensitivity / epsilon
    # 每个元素都是从拉普拉斯分布中独立采样的，位置参数为0，比例参数为noise_b
    noise = np.random.laplace(loc=0, scale=noise_b, size=1)
    # pdb.set_trace()
    return score + noise


def Exp_noise(score, epsilon, sensitivity):
    lambda_param = 2 * sensitivity / epsilon
    noise = np.random.exponential(scale=lambda_param, size=1)
    return score + noise


def add_noise_adaptive(embeddings, noise_factor, mean=0, std=1, noise_factors=[]):
    batch, seq, hidden = embeddings.shape
    noise_factors_expanded = noise_factors.unsqueeze(-1).expand(batch, seq, hidden).to(dtype=embeddings.dtype)
    noise = (noise_factor * noise_factors_expanded * torch.normal(mean=mean, std=std, size=embeddings.shape)
             .to(embeddings.device, dtype=embeddings.dtype))

    mask = noise_factors_expanded == 1
    embeddings = torch.where(mask, 0.5 * noise + 0.5 * embeddings, embeddings)
    mask = noise_factors_expanded == 0
    embeddings = torch.where(mask, 0.15 * noise + 0.85 * embeddings, embeddings)
    print("Adding gaussian noise stably!!")
    return embeddings


def add_gauss_noise_adaptive(embeddings, noise_factor, mean=0, std=1, prompt_masks=[]):
    batch, seq, hidden = embeddings.shape
    prompt_masks = prompt_masks.unsqueeze(-1).expand(batch, seq, hidden).to(dtype=embeddings.dtype)
    # 计算每个序列的prompt中心
    # batch_1_dimension
    prompt_centers = (embeddings * prompt_masks).sum(dim=1) / (prompt_masks.sum(dim=1).to(dtype=embeddings.dtype))
    # 计算欧式距离, 距离越远，噪声因子越大
    # pdb.set_trace()
    distances = torch.norm(embeddings - prompt_centers.unsqueeze(1), dim=-1)
    noise_factors = (distances - distances.min(dim=-1, keepdim=True)[0]) / \
                    (distances.max(dim=-1, keepdim=True)[0] - distances.min(dim=-1, keepdim=True)[0])


    noise_factors_expanded = noise_factors.unsqueeze(-1).expand(batch, seq, hidden).to(dtype=embeddings.dtype)
    noise_factors_expanded = torch.where(prompt_masks == 1, noise_factors_expanded, 0)
    # 生成高斯噪声
    noise = (noise_factor * torch.normal(mean=mean, std=std, size=embeddings.shape)
             .to(embeddings.device, dtype=embeddings.dtype))

    # 添加噪声到embeddings
    noisy_embeddings = embeddings + noise_factors_expanded * noise
    print("Adding gaussian noise adaptively!")
    return noisy_embeddings


def normalize_and_add_noise(embeddings, noise_factor, noise_factors):
    batch_mean = torch.mean(embeddings, dim=1, keepdim=True)
    batch_std = torch.std(embeddings, dim=1, keepdim=True)
    # normalized_embedding = (embedding - batch_mean) / batch_std

    embedding_with_noise = add_noise_adaptive(embeddings, noise_factor=noise_factor, noise_factors=noise_factors)
    nbatch_mean = torch.mean(embedding_with_noise)
    nbatch_std = torch.std(embedding_with_noise)
    normalized_embedding_with_noise = (embedding_with_noise - nbatch_mean) / nbatch_std

    embedding_with_noise = normalized_embedding_with_noise * batch_std + batch_mean
    return embedding_with_noise
