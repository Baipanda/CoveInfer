"""
    by bjy
    这段代码定义了一个支持 RNM 消毒的数据集
    在 token 层面 对低频（敏感）词进行 Report-Noisy-Max 替换，
    从而在 不修改 embedding、不回 TEE 解密 的前提下，实现 LDP。

    这段代码大概率是用来生成 low_freq_words 的
"""
import json
import pdb
import random
import torch
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
import random
import os
from utils import DpConfig, Lap_noise, Gas_noise
import numpy as np

# 设置随机数种子
random.seed(52)


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0], dtype=torch.long)
    attention_mask = torch.tensor(batch[1], dtype=torch.float16)
    labels = torch.tensor(batch[2], dtype=torch.long)
    noise_factors = torch.tensor(batch[3], dtype=torch.float16)

    return input_ids, attention_mask, labels, noise_factors


# 读取原始数据
class DataHelper:
    def __init__(self, data_path, val_set_size):
        self.data_path = data_path
        self.val_set_size = val_set_size

    def load_data(self):
        results = []
        with open(self.data_path, "r") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    def gen_data(self):
        data = self.load_data()
        random.shuffle(data)
        train_data = data[:-self.val_set_size]
        valid_data = data[-self.val_set_size:]
        return train_data, valid_data


class LlamaDataset(Dataset):
    def __init__(self, tokenizer, data, data_index):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"
        self.data = data
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.pad_token_id = 0
        else:
            self.pad_token_id = self.tokenizer.pad_token_id  # 0
        self.bos_token_id = self.tokenizer.bos_token_id  # 1
        self.eos_token = self.tokenizer.eos_token
        self.bos_token = self.tokenizer.bos_token

        self.data_index = data_index

        if self.data_index == 0:
            self.inputs_len = 150
            self.answer_len = 16
        elif self.data_index == 1:
            self.inputs_len = 300
            self.answer_len = 64
        elif self.data_index == 2:
            self.inputs_len = 64
            self.answer_len = 16

        self.label_pad_token_id = -100

        self.low_freq_words_path_list = ['../../Datasets/MedQA/questions/US/low_freq_words.txt',
                                         '../../Datasets/DialogSum/low_freq_words.txt',
                                         '../../Datasets/SST-2/low_freq_words.txt']

        self.low_freq_words_path = self.low_freq_words_path_list[self.data_index]
        self.low_freq_words = []
        self.nearest_tokens = {}

        with open(self.low_freq_words_path, 'r') as f:
            l = f.readlines()
            for item in l:
                self.low_freq_words.append(item.strip)

        with open("../../Models/Llama-2-7b-chat-hf/nearest_tokens_100.json", "r") as f:
            self.nearest_tokens = json.load(f)

    def report_private_tokens(self, input_ids, prompt_masks):
        sensitivity = DpConfig.sensitivity
        epsilon = DpConfig.lap_epsilon

        seq_len = len(input_ids)
        # print(input_ids)
        # print(seq_len)
        private_tokens = [0] * seq_len

        for j in range(seq_len):
            # pdb.set_trace()
            # print(j)
            if prompt_masks[j] == 1:
                private_tokens[j] = input_ids[j]
            elif prompt_masks[j] == 0 and input_ids[j] in self.low_freq_words:
                x = str(input_ids[j])
                # 获取词表中最近的100个单词的距离
                R = [item[0] for item in self.nearest_tokens[x]]
                distances = [item[1] for item in self.nearest_tokens[x]]
                # 为每个距离添加噪声：拉普拉斯噪声或者矩阵高斯噪声
                if DpConfig.noise_type == 0:
                    noisy_distances = [Gas_noise(distance) for distance in distances]
                else:
                    noisy_distances = [Lap_noise(distance, epsilon, sensitivity) for distance in distances]
                # 找到最大分数对应的token id
                private_tokens[j] = R[np.argmax(noisy_distances)]
            elif prompt_masks[j] == 0 and input_ids[j] not in self.low_freq_words:
                if random.random() < 0.3:
                    x = str(input_ids[j])
                    # 获取词表中最近的100个单词的距离
                    R = [item[0] for item in self.nearest_tokens[x]]
                    distances = [item[1] for item in self.nearest_tokens[x]]
                    if DpConfig.noise_type == 0:
                        noisy_distances = [Gas_noise(distance) for distance in distances]
                    else:
                        # 为每个距离添加噪声：拉普拉斯噪声或者矩阵高斯噪声
                        noisy_distances = [Lap_noise(distance, epsilon, sensitivity) for distance in distances]
                    # 找到最近距离对应的token id
                    private_tokens[j] = R[np.argmax(noisy_distances)]
                else:
                    private_tokens[j] = input_ids[j]

        return private_tokens

    def count_and_save_low_freq_words(self):
        low_freq_token_count = {}
        for data_point in self.data:
            if self.data_index == 0:
                text = data_point["question"] + " " + " ".join(data_point["options"].values()) + " " + data_point[
                    "answer"]
            elif self.data_index == 1:
                text = data_point["dialogue"] + " " + data_point["summary"]
            elif self.data_index == 2:
                text = data_point["input"]

            tokens = self.tokenizer.encode(text)
            for token in tokens:
                if token not in low_freq_token_count:
                    low_freq_token_count[token] = 1
                low_freq_token_count[token] += 1

        # 按照出现次数升序排序
        sorted_low_freq_tokens = sorted(low_freq_token_count.items(), key=lambda x: x[1])
        pdb.set_trace()
        # 计算要保存的token数量
        num_tokens_to_save = int(32000 * self.freq_level)

        # 将前num_tokens_to_save个token保存到txt文件中
        with open(self.low_freq_words_path, "w") as f:
            for word, count in sorted_low_freq_tokens[:num_tokens_to_save]:
                f.write(f"{word}\n")

        print(f"低频敏感词已保存到: {self.low_freq_words_path}")

    def generate_medqa_prompt(self, elem):
        instruction = "You are a doctor. According to the question, make a choice. Q: "
        sentence = elem["question"]
        for key, value in elem["options"].items():
            sentence = sentence + '\n' + key + ': ' + value
        answer = "The Answer is: " + elem['answer_idx'] + ": " + elem["answer"]
        # print(f"sentence is {sentence}")
        return instruction, sentence, answer

    def generate_dialogsum_prompt(self, elem):
        instruction = (
            "Summarize the dialogue between two speakers, focusing on the most salient information discussed. "
            "The summary should be concise while preserving important named entities mentioned,  "
            "not exceeding 256 characters, using a third-person perspective with formal language."
            "Dialogue: \n")
        input = elem["dialogue"]
        summary = "Summary: " + elem["summary"]
        return instruction, input, summary

    def generate_sst2_prompt(self, elem):
        instruction = ("Classify the sentiment of the given movie review text into Positive or Negative. "
                       "Movie Review:")
        input = elem["input"]
        output = "This movie review is " + elem["label"]
        return instruction, input, output

    def tokenize(self, prompt, inputs, answer):
        prompt_t = self.tokenizer(
            prompt,
            padding=False,
            add_special_tokens=False,
            return_tensors=None
        )
        inputs_t = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.inputs_len,
            padding=False,
            add_special_tokens=False,
            return_tensors=None
        )
        answer_t = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.answer_len - 2,
            padding=False,
            add_special_tokens=False,
            return_tensors=None
        )
        prompt_len = len(prompt_t['input_ids'])
        input_len = len(inputs_t["input_ids"])
        answer_len = len(answer_t["input_ids"])

        # print(f"prompt len is {prompt_len}")
        # print(f"input len is {input_len}")
        # print(f"answer len is {answer_len}")

        input_id = (prompt_t["input_ids"] + inputs_t["input_ids"] + [self.bos_token_id] + answer_t["input_ids"] +
                    [self.eos_token_id])

        attention_mask = (prompt_t["attention_mask"] + inputs_t["attention_mask"] + [1] + answer_t["attention_mask"] +
                          [1])

        max_length = self.inputs_len + self.answer_len + 100
        pad_len = max_length - len(input_id)
        # print(f"pad_len is {pad_len}")
        # 左边填充
        input_ids = [self.pad_token_id] * pad_len + input_id
        attention_mask = [0] * pad_len + attention_mask
        labels = [self.label_pad_token_id] * pad_len + input_id
        prompt_mask = [1] * pad_len + [1] * prompt_len + [0] * (len(input_ids) - prompt_len - pad_len)

        return input_ids, attention_mask, labels, prompt_mask

    def generate_and_tokenize_prompt(self, data_point):
        if self.data_index == 0:
            prompt, inputs, answer = self.generate_medqa_prompt(data_point)
        elif self.data_index == 1:
            prompt, inputs, answer = self.generate_dialogsum_prompt(data_point)
        elif self.data_index == 2:
            prompt, inputs, answer = self.generate_sst2_prompt(data_point)
        # print(f"prompt is {prompt}")
        # print(f"answer is {answer}")
        input_ids, attention_mask, labels, prompt_mask = self.tokenize(prompt, inputs, answer)

        if DpConfig.emb_add_noise:
            input_ids = self.report_private_tokens(input_ids, prompt_mask)

        return input_ids, attention_mask, labels, prompt_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        input_ids, attention_mask, labels, prompt_mask = self.generate_and_tokenize_prompt(data_point)
        return input_ids, attention_mask, labels, prompt_mask
