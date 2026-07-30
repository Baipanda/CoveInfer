import json
import random
import torch
import numpy as np
import time
from utils import DpConfig, Lap_noise, Gas_noise
from transformers import GenerationConfig, LlamaTokenizer, AutoTokenizer
from modeling_llama import LlamaForCausalLM


class ChatGenerator:
    def __init__(self):
        # =========================
        # 基本配置
        # =========================
        self.model_index = 1
        self.model_path_list = [
            "../../Models/Llama-3-8B-Instruct-hf",
            "../../Models/Llama-2-7b-chat-hf"
        ]
        self.model_path = self.model_path_list[self.model_index]

        self.max_new_tokens = 1024
        self.device = torch.device("cuda:0")

        # =========================
        # DP 相关
        # =========================
        self.lap_epsilon = 100.0
        # 低频词文件
        self.low_freq_words_path = "../../Datasets/DialogSum/low_freq_words.txt"
        self.low_freq_words = []
        with open(self.low_freq_words_path, "r") as f:
            for line in f:
                self.low_freq_words.append(line.strip())
        # 距离最近的30个替换词列表
        with open("../../Models/Llama-2-7b-chat-hf/nearest_tokens_30.json", "r") as f:
            self.nearest_tokens = json.load(f)

        # =========================
        # tokenizer & model
        # =========================
        if self.model_index == 0:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "64GiB", "cpu": "64GiB"},
        )

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = True
        self.model.eval()

        # generation config
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            do_sample=True,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # =========================
        # 性能统计相关
        # =========================
        self.stats = {
            "total_tokens_generated": 0,
            "total_time_seconds": 0.0,
            "total_queries": 0,
            "dp_processing_time": 0.0
        }

    # ======================================================
    # DP token 替换
    # ======================================================
    def report_private_tokens(self, input_ids):
        dp_start_time = time.time()

        sensitivity = DpConfig.sensitivity
        epsilon = self.lap_epsilon
        seq_len = len(input_ids)

        private_tokens = [0] * seq_len
        for j in range(seq_len):
            # 如果当前token如果属于低频词
            if input_ids[j] in self.low_freq_words:
                x = str(input_ids[j])
                # R 是30个邻近词的 id
                R = [item[0] for item in self.nearest_tokens[x]]
                distances = [item[1] for item in self.nearest_tokens[x]]

                if DpConfig.noise_type == 0:
                    noisy_distances = [Gas_noise(d) for d in distances]
                else:
                    noisy_distances = [
                        Lap_noise(d, epsilon, sensitivity) for d in distances
                    ]
                # 取了一个分数最大的
                private_tokens[j] = R[np.argmax(noisy_distances)]
            else:
                if random.random() < 0.3:
                    x = str(input_ids[j])
                    R = [item[0] for item in self.nearest_tokens[x]]
                    distances = [item[1] for item in self.nearest_tokens[x]]

                    if DpConfig.noise_type == 0:
                        noisy_distances = [Gas_noise(d) for d in distances]
                    else:
                        noisy_distances = [
                            Lap_noise(d, epsilon, sensitivity) for d in distances
                        ]

                    private_tokens[j] = R[np.argmax(noisy_distances)]
                else:
                    private_tokens[j] = input_ids[j]

        dp_processing_time = time.time() - dp_start_time
        self.stats["dp_processing_time"] += dp_processing_time

        return private_tokens

    # ======================================================
    # 单轮聊天生成（带性能监控）
    # ======================================================
    def generate_chat(self, user_input, history=""):
        self.stats["total_queries"] += 1

        prompt = (
            user_input
        )

        # ========= DP token noise =========
        if DpConfig.emb_add_noise:
            tokenized = self.tokenizer(user_input, return_tensors=None)
            private_ids = self.report_private_tokens(tokenized["input_ids"])
            # 转换为可读的文本
            private_input = self.tokenizer.decode(
                torch.tensor(private_ids),
                skip_special_tokens=True
            )

            final_prompt = (
                private_input
            )

            print("\n===== final_prompt =====")
            print(final_prompt)
            print("========================\n")
        else:
            final_prompt = prompt

        inputs = self.tokenizer(
            final_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]
        # 计算推理时间的起点
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config,
                max_new_tokens=self.max_new_tokens,
            )
        inference_time = time.time() - start_time

        output_length = output.shape[1]
        tokens_generated = output_length - input_length

        self.stats["total_tokens_generated"] += tokens_generated
        self.stats["total_time_seconds"] += inference_time

        decoded = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        if "Assistant:" in decoded:
            answer = decoded.rsplit("Assistant:", 1)[-1].strip()
        else:
            answer = decoded.strip()

        print(f"\n本次推理统计:")
        print(f"  输入Token数: {input_length}")
        print(f"  生成Token数: {tokens_generated}")
        print(f"  推理时间: {inference_time:.2f}秒")
        print(
            f"  速度: {tokens_generated / inference_time:.2f} tokens/秒"
            if inference_time > 0 else "  速度: N/A"
        )

        return answer


# ======================================================
# 命令行 Chat
# ======================================================
def chat_loop():
    generator = ChatGenerator()
    history = ""

    print("DP-Chat started (输入 exit / quit 退出)")
    print("输入 'clear' 清空历史\n")

    while True:
        user_input = input("\nUser: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nBye")
            break

        if user_input.lower() == "clear":
            history = ""
            print("历史已清空")
            continue

        response = generator.generate_chat(user_input, history)
        print(f"\nResponse: {response}")

        history += f"User: {user_input}\nAssistant: {response}\n"


if __name__ == "__main__":
    chat_loop()
