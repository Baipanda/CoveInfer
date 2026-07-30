"""
    by bjy
                CMIF 核心思想
                    │
        ┌───────────┴───────────┐
        │                       │
   BERT 分类任务           LLaMA 生成任务
   token-level RNM     token + embedding DP
        │                       │
  这份代码              你刚才的 train.py

"""
import pdb
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
from utils import DpConfig, Lap_noise, Gas_noise
import numpy as np


class BertClassifier:
    def __init__(self, model_name, num_labels, device, epochs=3, batch_size=128, lr=2e-5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lap_epsilon = 3.0

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(device)

        self.low_freq_words = []
        self.nearest_tokens = {}
        # '../../Datasets/SST-2/low_freq_words.txt'
        self.low_freq_words_path = "../../Datasets/QNLI/low_freq_words.txt"

        with open(self.low_freq_words_path, 'r') as f:
            l = f.readlines()
            for item in l:
                self.low_freq_words.append(item.strip)

        with open("../../Models/bert-base-uncased/nearest_tokens_30.json", "r") as f:
            self.nearest_tokens = json.load(f)

    def count_and_save_low_freq_words(self, data_file):
        low_freq_token_count = {}
        with open(data_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                text = data["text1"] + data["text2"]
                tokens = self.tokenizer.encode(text)
                for token in tokens:
                    if token not in low_freq_token_count:
                        low_freq_token_count[token] = 1
                    low_freq_token_count[token] += 1
        # 按照出现次数升序排序
        sorted_low_freq_tokens = sorted(low_freq_token_count.items(), key=lambda x: x[1])
        # 计算要保存的token数量
        num_tokens_to_save = int(30253 * 0.2)

        # 将前num_tokens_to_save个token保存到txt文件中
        with open(self.low_freq_words_path, "w") as f:
            for word, count in sorted_low_freq_tokens[:num_tokens_to_save]:
                f.write(f"{word}\n")

        print(f"低频敏感词已保存到: {self.low_freq_words_path}")

    def report_private_tokens(self, input_ids):
        sensitivity = 1.0
        epsilon = self.lap_epsilon
        num = len(input_ids)

        seq_len = len(input_ids[0])
        # print(seq_len)
        private_tokens = [[0] * seq_len for _ in range(num)]

        for i in range(num):
            for j in range(seq_len):
                # pdb.set_trace()
                # print(j)
                if input_ids[i][j] != 0 and input_ids[i][j] in self.low_freq_words:
                    x = str(input_ids[i][j])
                    R = [item[0] for item in self.nearest_tokens[x]]
                    distances = [item[1] for item in self.nearest_tokens[x]]
                    # 为每个距离添加噪声：拉普拉斯噪声或者矩阵高斯噪声
                    if DpConfig.noise_type == 0:
                        noisy_distances = [Gas_noise(distance) for distance in distances]
                    else:
                        noisy_distances = [Lap_noise(distance, epsilon, sensitivity) for distance in distances]
                    # 找到最大分数对应的token id
                    private_tokens[i][j] = R[np.argmax(noisy_distances)]
                else:
                    if random.random() < 0.3:
                        x = str(input_ids[i][j])
                        # 获取词表中最近的100个单词的距离
                        R = [item[0] for item in self.nearest_tokens[x]]
                        distances = [item[1] for item in self.nearest_tokens[x]]
                        if DpConfig.noise_type == 0:
                            noisy_distances = [Gas_noise(distance) for distance in distances]
                        else:
                            # 为每个距离添加噪声：拉普拉斯噪声或者矩阵高斯噪声
                            noisy_distances = [Lap_noise(distance, epsilon, sensitivity) for distance in distances]
                        # 找到最大分数对应的token id
                        private_tokens[i][j] = R[np.argmax(noisy_distances)]
                    else:
                        private_tokens[i][j] = input_ids[i][j]

        return private_tokens

    def train(self, train_texts, train_labels):
        train_encodings = self.tokenizer(train_texts, max_length=128, truncation=True, padding=True,
                                         return_tensors=None)
        pdb.set_trace()
        input_ids = self.report_private_tokens(train_encodings['input_ids'])
        train_dataset = TensorDataset(
            torch.tensor(input_ids),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(train_labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for batch in tqdm(train_loader):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def evaluate(self, test_texts, test_labels):
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True).to(self.device)

        test_inputs = self.report_private_tokens(test_encodings['input_ids'])

        test_dataset = TensorDataset(
            torch.tensor(test_inputs),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(test_labels)
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds.extend(outputs.logits.argmax(dim=1).tolist())

        return preds


def parse_sst_json_data(data_file):
    texts = []
    labels = []

    with open(data_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data['input']
            label = data['label']

            # 将标签转换为数字
            if label == 'positive':
                label = 1
            elif label == 'negative':
                label = 0
            else:
                raise ValueError(f"Invalid label: {label}")

            texts.append(text)
            labels.append(label)

    return texts, labels


def parse_qnli_json_data(data_file):
    texts = []
    labels = []

    with open(data_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = "text1: " + data["text1"] + "text2: " + data["text2"]
            label = data['label']

            texts.append(text)
            labels.append(label)

    return texts, labels


if __name__ == "__main__":
    train_file = '/Datasets/QNLI/train.jsonl'
    test_file = '/Datasets/QNLI/test.jsonl'
    # from modelscope.msdatasets import MsDataset

    train_texts, train_labels = parse_qnli_json_data(train_file)
    test_texts, test_labels = parse_qnli_json_data(test_file)

    classifier = BertClassifier("/Models/bert-base-uncased", num_labels=2,
                                device=torch.device('cuda:0'))
    # classifier.count_and_save_low_freq_words(train_file)

    classifier.train(train_texts, train_labels)
    predictions = classifier.evaluate(test_texts, test_labels)

    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)  # 添加计算准确率

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")  # 输出准确率
