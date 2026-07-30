"""
    by bjy
    测试文件
    主要函数：report_private_tokens
    1.  查 embedding 最近邻（nearest_tokens）
        对距离加 Laplace / Gaussian DP 噪声
        选噪声后“最优”的 token 替换
    2.  以 30% 概率随机扰动
        否则保持原 token
    测试完整的流程：evaluateSum()

    utils.py
       ↓
    data_helper.py
       ↓
    train.py   （训练 / 微调，生成模型）
       ↓
    （保存 LoRA / 模型权重）
       ↓
    generator / test.py   （推理 + 评测）

        Trainer.__init__()
      ├─ get_data_loader()     ← data_helper.py
      ├─ tokenizer 加载
      └─ 训练参数初始化

    Trainer.LoadModelWeight()
      ├─ 加载 LLaMA
      ├─ 注入 LoRA
      └─ 开启 gradient checkpoint

    Trainer.train()
      ├─ for epoch
      │    ├─ for batch
      │    │    ├─ model(...)
      │    │    │    ↳ embedding / logits（含 DP）
      │    │    ├─ EMOLoss
      │    │    └─ backward
      │    └─ eval()
      └─ 保存 LoRA 权重
"""
import json
import pdb
import sys
import os
import random
from utils import DpConfig, Lap_noise, Gas_noise
import torch
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, AutoTokenizer
from modeling_llama import LlamaForCausalLM
from peft import PeftModel
from datasets import load_metric

metric = load_metric("rouge.py", trust_remote_code=True)


def mean(item: list) -> float:
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


class Generator:
    def __init__(self):
        self.data_index = 1
        self.model_index = 1
        self.max_len = [16, 64, 16, 50]
        self.model_path_list = ["../../Models/Llama-2-7b-chat-hf",
                                "../../Models/Llama-3-8B-Instruct-hf"]
        self.model_path = self.model_path_list[self.model_index]
        self.result_path = ""

        self.low_freq_words_path_list = ['../../Datasets/MedQA/questions/US/low_freq_words.txt',
                                         '../../Datasets/DialogSum/low_freq_words.txt',
                                         '../../Datasets/SST-2/low_freq_words.txt',
                                         '../../Datasets/DialogSum/low_freq_words.txt']

        self.low_freq_words_path = self.low_freq_words_path_list[self.data_index]
        self.low_freq_words = []
        self.nearest_tokens = {}

        with open(self.low_freq_words_path, 'r') as f:
            l = f.readlines()
            for item in l:
                self.low_freq_words.append(item.strip)

        with open("../../Models/Llama-3-8B-Instruct-hf/nearest_tokens_30.json", "r") as f:
            self.nearest_tokens = json.load(f)

        if self.model_index == 0:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
        )
        self.device = torch.device("cuda:0")

        self.lora = True
        self.lap_epsilon = 100.0
        if self.model_index == 0:
            self.lora_model_path = ("../../Models_with_DP/Llama3-8b/DialogSum_" + str(self.lap_epsilon) +
                                    "/1870")

        self.model.to(self.device)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = False
        self.model.eval()
        self.answer_n = 1

    def generate_mdeqa_prompt(self, elem):
        instruction = "You are a doctor. According to the question, make a choice. Q: "
        input = elem["question"]
        for key, value in elem["options"].items():
            input = input + ' ' + key + ': ' + value + "The Answer is: "
        label = elem['answer_idx'] + ": " + elem["answer"]
        # print(f"sentence is {sentence}")
        return instruction, input, label

    def generate_dialogsum_prompt(self, elem):
        instruction = (
            "Summarize the dialogue between two speakers, focusing on the most salient information discussed. "
            "The summary should be concise while preserving important named entities mentioned,  "
            "not exceeding 256 characters, using a third-person perspective with formal language."
            "Dialogue: \n")
        input = elem["dialogue"]
        label = "Summary: " + elem["summary%d" % TEST_SUMMARY_ID]
        return instruction, input, label

    def generate_sst2_prompt(self, elem):
        instruction = ("Classify the sentiment of the given movie review text into Positive or Negative. "
                       "Movie Review:")
        input = elem["input"]
        label = elem["label"]
        return instruction, input, label

    def generate_ifeval_prompt(self, elem):
        prompt = elem["prompt"]
        return prompt

    def report_private_tokens(self, input_ids):
        sensitivity = DpConfig.sensitivity
        epsilon = self.lap_epsilon
        seq_len = len(input_ids)
        # print(seq_len)
        private_tokens = [0] * seq_len

        for j in range(seq_len):
            # pdb.set_trace()
            # print(j)
            if input_ids[j] in self.low_freq_words:
                x = str(input_ids[j])
                R = [item[0] for item in self.nearest_tokens[x]]
                distances = [item[1] for item in self.nearest_tokens[x]]
                # 为每个距离添加噪声：拉普拉斯噪声或者矩阵高斯噪声
                if DpConfig.noise_type == 0:
                    noisy_distances = [Gas_noise(distance) for distance in distances]
                else:
                    noisy_distances = [Lap_noise(distance, epsilon, sensitivity) for distance in distances]
                # 找到最大分数对应的token id
                private_tokens[j] = R[np.argmax(noisy_distances)]
            else:
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
                    # 找到最大分数对应的token id
                    private_tokens[j] = R[np.argmax(noisy_distances)]
                else:
                    private_tokens[j] = input_ids[j]

        return private_tokens

    def evaluateSum(self, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=5, **kwargs):
        if self.data_index == 0:
            prompt, input, label = self.generate_mdeqa_prompt(input)
        elif self.data_index == 1:
            prompt, input, label = self.generate_dialogsum_prompt(input)
        elif self.data_index == 2:
            prompt, input, label = self.generate_sst2_prompt(input)
        elif self.data_index == 3:
            prompt = self.generate_ifeval_prompt(input)
            input = ""

        if DpConfig.emb_add_noise:
            inputs = self.tokenizer(input, return_tensors=None)
            # pdb.set_trace()
            private_inputs_ids = self.report_private_tokens(inputs["input_ids"])
            private_input = self.tokenizer.decode(torch.tensor(private_inputs_ids))
            # pdb.set_trace()
            inputs = (
                self.tokenizer(prompt + private_input + "Summary: ", padding=False, add_special_tokens=False, return_tensors="pt")
                .to(self.device))
        else:
            inputs = (self.tokenizer(prompt + input + "Summary: ", padding=False, add_special_tokens=False, return_tensors="pt")
                      .to(self.device))

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        # res = {"Prompt": prompt, "Results": []}
        for i in range(self.answer_n):
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=self.max_len[self.data_index],
                )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s, skip_special_tokens=True).replace("\n", " ")

            answer = {"prompt": prompt, "response": output}

            with open(self.result_path, "a+", encoding="utf-8") as fw:
                # json.dump(answer, fw, ensure_ascii=False)
                # fw.write("\n")
                fw.write(f" Output: {output}\n")
                fw.write(f" Label: {label}\n")
            # output = output.split(prompt+private_inputs)[-1]

            # res["Results"].append(output)
            # print(f"Prompt is {prompt}\n")
            print(f"Response is {output}\n")
            # acc = accuracy(output, label)

        # with open(self.result_path, "a+") as fw:
        #     json.dump(res, fw)

        if self.data_index == 3:
            return

        # label_id = self.tokenizer(label)
        # label_id = torch.tensor(label_id["input_ids"], dtype=torch.long, device=self.device)
        # label_vector = self.model.model.embed_tokens(label_id)
        # label_vector = self.model.base_model.model.model.embed_tokens(label_id)
        # pre_vector = self.model.base_model.model.model.embed_tokens(s)
        # # pre_vector = self.model.model.embed_tokens(s)
        # cs = compute_cs(label_vector, pre_vector)
        # print(f"cs is {cs}")

        result = "Summary: " + output.split("Summary: ")[-1]
        # result = output.split(prompt)[-1]
        rouge_score = compute_rouges(result, label, self.tokenizer)

        # if not os.path.exists(self.result_path):
        #     os.mkdir(self.result_path)
        cs = 0
        return cs, rouge_score


def compute_rouges(predictions, labels, tokenizer):
    # predictions = predictions.split()
    # labels = labels.split()
    result = metric.compute(predictions=[predictions], references=[labels], use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    # 将所有指标值四舍五入到小数点后4位, 并返回结果字典。
    return {k: round(v, 4) for k, v in result.items()}


def compute_cs(label_v, pre_v):
    # 获取两个序列的长度
    label_len = label_v.shape[0]
    pre_len = pre_v.shape[0]

    # 确定需要填充的长度
    pad_len = max(label_len, pre_len)

    # 对较短的序列进行填充
    if label_len < pad_len:
        label_v = torch.cat([label_v, torch.zeros(pad_len - label_len, label_v.shape[1]).to(label_v.device)], dim=0)
    if pre_len < pad_len:
        pre_v = torch.cat([pre_v, torch.zeros(pad_len - pre_len, pre_v.shape[1]).to(pre_v.device)], dim=0)

    # 计算两个张量的点积
    dot_product = torch.sum(label_v * pre_v, dim=1)

    # 计算两个张量的范数
    label_norm = torch.norm(label_v, dim=1)
    pre_norm = torch.norm(pre_v, dim=1)

    # 计算余弦相似度
    cs = dot_product / (label_norm * pre_norm + 1e-8)

    # 返回余弦相似度的均值
    return torch.mean(cs)


def accuracy(pred_word, true_word):
    pred_word = pred_word.lower()
    true_word = true_word.lower()
    if true_word in pred_word:
        return 1
    else:
        return 0


if __name__ == "__main__":
    generator = Generator()

    test_path_list = ["../../Datasets/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl",
                      "../../Datasets/DialogSum/dialogsum.test.jsonl",
                      "../../Datasets/SST-2/test.jsonl",
                      "../../Datasets/IFEVAL/data/input_data.jsonl"]
    test_path = test_path_list[generator.data_index]
    if generator.data_index == 0:
        generator.result_path = ("./test/" + "Llama" + str(generator.model_index + 2) + "_medqa_" +
                                 str(generator.lap_epsilon) + ".txt")
    elif generator.data_index == 1:
        generator.result_path = ("./test/" + "Llama" + str(generator.model_index + 2) + "_dialog_" +
                                 str(generator.lap_epsilon) + ".txt")
    elif generator.data_index == 2:
        generator.result_path = ("./test/" + "Llama" + str(generator.model_index + 2) + "_sst2_" +
                                 str(generator.lap_epsilon) + ".txt")
    elif generator.data_index == 3:
        generator.result_path = ("./test/" + "Llama" + str(generator.model_index + 2) + "_IFEVAL_" +
                                 str(generator.lap_epsilon) + ".jsonl")

    TEST_SUMMARY_ID = 1

    rouge1 = []
    rouge2 = []
    rougeL = []
    cs_list = []

    count = 0
    with open(test_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='test', unit='line'):
            json_str = line.strip()
            inputs = json.loads(json_str)
            if generator.data_index != 3:
                cs, result = generator.evaluateSum(input=inputs)
                rouge1.append(result['rouge1'])
                rouge2.append(result['rouge2'])
                rougeL.append(result['rougeL'])
                cs_list.append(cs)
            else:
                generator.evaluateSum(input=inputs)
            count += 1
            # if count >= 500:
            #     break

    if generator.data_index != 3:
        print(f"rouge1 mean is {mean(rouge1):.2f}")
        print(f"rouge2 mean is {mean(rouge2):.2f}")
        print(f"rougeL mean is {mean(rougeL):.2f}")
        print(f"cs mean is {mean(cs_list)}")
