"""
    by bjy
    dp_noise 差分隐私噪声
"""
import os
import math
import pdb
import numpy as np
from transformers import BertForMaskedLM, LlamaForCausalLM, BertForSequenceClassification
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
from prv_accountant import Accountant
import json
import torch
from torch.nn.functional import pdist


# 矩阵高斯噪声
# Matrix Gaussian Noise
def matrix_gaussian_noise(epsilon, delta, sensitivity):
    # 计算标准正态分布的累积分布函数CDF
    def function_phi(t):
        return (1 + math.erf(t / math.sqrt(2))) / 2

    # 用于确定噪声的上下界
    def B_plus_function(v, epsilon):
        return function_phi(math.sqrt(epsilon * v)) - math.exp(epsilon) * function_phi(-math.sqrt(epsilon * (v + 2)))

    def B_minus_function(u, epsilon):
        return function_phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * function_phi(-math.sqrt(epsilon * (u + 2)))

    # 计算最优的R值
    def compute_R(epsilon, delta, iterations=5000):
        delta_0 = function_phi(0) - math.exp(epsilon) * function_phi(-math.sqrt(2 * epsilon))
        # print(delta_0)

        start, end = 0, 1e5

        B_function = B_plus_function if delta >= delta_0 else B_minus_function
        for i in range(iterations):
            mid = (start + end) / 2
            value = B_function(mid, epsilon)
            if value < delta:
                end = mid
            else:
                start = mid

        u_star = end
        b_value = B_function(end, epsilon)

        if delta >= delta_0:
            alpha = math.sqrt(1 + u_star / 2) - math.sqrt(u_star / 2)
        else:
            alpha = math.sqrt(1 + u_star / 2) + math.sqrt(u_star / 2)

        R = math.sqrt(2 * epsilon) / alpha
        return R

    R = compute_R(epsilon, delta)
    noise_b = sensitivity / R
    # noise_matrix = noise_b * np.random.normal(size=dim)
    return noise_b


# 生成满足 ε-差分隐私的噪声矩阵。
# 输入参数包括隐私预算epsilon、矩阵维度dim和敏感度sensitivity
def Lap_noise(epsilon, dim, sensitivity):
    noise_b = sensitivity / epsilon
    # 每个元素都是从拉普拉斯分布中独立采样的，位置参数为0，比例参数为noise_b
    noise_matrix = np.random.laplace(loc=0, scale=noise_b, size=dim)
    return noise_matrix


"""生成满足 (ε, δ)-差分隐私的噪声。
输入参数包括隐私预算epsilon、delta和敏感度sensitivity。
函数首先根据给定的epsilon和delta计算高斯分布的标准差sigma，公式为sqrt(2 * ln(1.25 / delta)) / epsilon。
然后，函数计算噪声参数noise_b，即高斯分布的标准差，通过将敏感度乘以sigma得到。
最后，函数返回计算得到的噪声参数noise_b"""


# https://en.m.wikipedia.org/wiki/Additive_noise_mechanisms#Gaussian_Mechanism
def Gaussian_noise(epsilon, delta, sensitivity):
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_b = sensitivity * sigma
    # noise_matrix = np.random.normal(loc=0,scale=noise_b,size=dim)
    return noise_b


# 计算满足差分隐私要求的多元高斯噪声的标准差
"""函数的输入参数包括：
epsilon：隐私预算参数，控制隐私保护的强度。
delta：隐私预算参数，表示隐私泄露的概率上界。
dim：嵌入向量的维度，通常是一个包含两个元素的列表或元组，表示嵌入矩阵的行数和列数。
sensitivity：嵌入矩阵的敏感度，表示在改变一条记录时，嵌入矩阵的最大变化量。
gamma：一个常数，用于控制噪声的大小和隐私保护的强度。
在嵌入矩阵中添加这个标准差的高斯噪声，可以保护个人隐私，防止敏感信息的泄露。
"""


# MVG@CCS18
def MVG_noise(epsilon, delta, dim, sensitivity, gamma):
    def cal_harmonic_number(r, m=1.0):
        sum = 0
        for i in range(1, r + 1):
            sum += 1 / i ** m
        return sum

    r = min(dim)
    H_r = cal_harmonic_number(r)
    H_r_half = cal_harmonic_number(r, m=0.5)
    alpha = (H_r + H_r_half) * gamma ** 2 + 2 * H_r * gamma * sensitivity
    zeta = 2 * math.sqrt(-dim[0] * dim[1] * math.log(delta)) - 2 * math.log(delta) + dim[0] * dim[1]
    beta = 2 * ((dim[0] * dim[1]) ** 0.25) * H_r * sensitivity * zeta
    noise_b = (2 * alpha * ((dim[0] * dim[1]) ** 0.25)) / (-beta + math.sqrt(beta ** 2 + 8 * alpha * epsilon))
    #
    # precisionBudget = ((-beta + np.sqrt(beta ** 2 + 8 * alpha * epsilon)) ** 2) / (4 * (alpha ** 2))
    return noise_b


def gamma_simulation(num_simulation=100000):
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    total_num = embedding_matrix.shape[0]
    total_indices = np.arange(total_num)
    norms = []
    for i in tqdm(range(num_simulation)):
        np.random.shuffle(total_indices)
        sampled_indices = total_indices[:512]
        embeddings = embedding_matrix[sampled_indices]
        norm2 = np.linalg.norm(embeddings)
        norms.append(norm2)
    print(max(norms))
    return max(norms)


# 通过蒙特卡洛模拟计算Llama嵌入矩阵的最大L2范数
def gamma_simulation_for_llama(num_simulation=100000):
    model = LlamaForCausalLM.from_pretrained("/home/yuhonglan/Models/Llama-2-7b-chat-hf")

    embedding_matrix = model.model.embed_tokens.weight.data
    # 将嵌入矩阵移动到GPU
    embedding_matrix = embedding_matrix.cuda()
    pdb.set_trace()
    # 词表大小
    total_num = embedding_matrix.shape[0]
    # 生成词表索引
    total_indices = torch.arange(total_num).cuda()
    norms = []
    for i in tqdm(range(num_simulation)):
        # 打乱词表
        shuffled_indices = total_indices[torch.randperm(total_num)]
        # 提取前2048个采样结果作为模拟
        sampled_indices = shuffled_indices[:4096]
        embeddings = embedding_matrix[sampled_indices]

        norm2 = torch.norm(embeddings).item()
        norms.append(norm2)
    print(f"max_norms is {max(norms)}")
    return max(norms)


def cal_sensitivity_bert_embedding():
    """
    This function may run up to a few minutes. The results are:
    L1 Sensitivity: 56.33
    L2 Sensitivity: 2.89
    """

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # l1-sensitivity
    # 使用cdist函数计算embedding_matrix中每对向量之间的曼哈顿距离（L1距离）。
    # 取所有距离中的最大值作为L1敏感度，并将其赋值给l1_sensitivity变量。
    distance = cdist(embedding_matrix, embedding_matrix, "cityblock")
    l1_sensitivity = np.max(distance)
    print(l1_sensitivity)

    # l2-sensitivity
    # 使用cdist函数计算embedding_matrix中每对向量之间的欧几里得距离（L2距离）。
    # 取所有距离中的最大值作为L2敏感度，并将其赋值给l2_sensitivity变量。
    distance = cdist(embedding_matrix, embedding_matrix, "euclidean")
    l2_sensitivity = np.max(distance)
    print(l2_sensitivity)

    return l1_sensitivity, l2_sensitivity


def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def get_nearest_tokens(model, vocab=[], batch_size=4000, top_k=30):
    num_tokens = 30522
    nearest_tokens = {}
    for i in range(0, num_tokens):
        vocab.append(i)
    # embedding_matrix = model.model.embed_tokens.weight.float()
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.float()

    for i in range(0, num_tokens, batch_size):
        batch_embeddings = embedding_matrix[i:i + batch_size]
        # 计算batch内的token与所有token之间的欧式距离
        # distances = torch.cdist(batch_embeddings, embedding_matrix, p=2)
        similarities = torch.matmul(batch_embeddings, embedding_matrix.transpose(0, 1))
        similarities_min = similarities.min()
        similarities_max = similarities.max()
        # 最大值-最小值归一化
        normalized_similarities = (similarities - similarities_min) / (similarities_max - similarities_min)
        # 对于batch内的每个token,找到最近的top_k个token
        nearest_indices = torch.argsort(normalized_similarities, dim=1, descending=True)[:, :top_k + 1]

        for j in range(len(batch_embeddings)):
            token = vocab[i + j]
            if token not in nearest_tokens:
                nearest_tokens[token] = {}
            for idx in nearest_indices[j]:
                near_token = vocab[idx]
                similarity = float(similarities[j][idx].item())
                nearest_tokens[token][near_token] = similarity

    final_nearest_tokens = {}
    for token, token_nearest in nearest_tokens.items():
        sorted_nearest = sorted(token_nearest.items(), key=lambda x: x[1], reverse=True)[1:top_k + 1]
        final_nearest_tokens[token] = sorted_nearest

    pdb.set_trace()
    save_nearest_tokens(final_nearest_tokens, "../../Models/bert-base-uncased/nearest_tokens_30.json")


def save_nearest_tokens(nearest_tokens, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nearest_tokens, f, ensure_ascii=False, indent=4)


def cal_sensitivity_llama_embedding(model):
    # pdb.set_trace()
    embedding_matrix = model.model.embed_tokens.weight.float().to("cuda:0")
    norm2 = torch.norm(embedding_matrix).item()
    print("norm2", norm2)
    # pdb.set_trace()
    batch_size = 2000
    num_batches = embedding_matrix.shape[0] // batch_size
    max_distance = 0

    for i in range(num_batches):
        print(f"step : {i}/{num_batches}")
        start = i * batch_size
        end = min((i + 1) * batch_size, embedding_matrix.shape[0])
        batch_matrix = embedding_matrix[start:end]

        # 计算批次内部的距离
        distance_batch_inner = pdist(batch_matrix, p=2)
        max_distance = max(max_distance, torch.max(distance_batch_inner).item())

        # 计算批次与之前所有向量的距离
        if i > 0:
            prev_matrix = embedding_matrix[:start]
            distance_batch_outer = torch.cdist(batch_matrix, prev_matrix, p=2)
            max_distance = max(max_distance, torch.max(distance_batch_outer).item())

    l2_sensitivity = max_distance
    return l2_sensitivity

def get_noise_multiplier(eps, delta, batch_size=8, dataset_size=50000, epoch=3, local_dp=False, noise_type='aGM'):
    # 如果是本地差分隐私
    if local_dp:
        if noise_type == 'aGM':  # 矩阵高斯噪声
            return matrix_gaussian_noise(epsilon=eps, delta=delta, sensitivity=49.04)
        elif noise_type == 'GM':  # 高斯噪声
            return Gaussian_noise(epsilon=eps, delta=delta, sensitivity=49.04)

    start_noise_multiplier = 0.2
    end_noise_multiplier = 50
    while True:
        mid_noise_multiplier = (start_noise_multiplier + end_noise_multiplier) / 2
        # 创建一个差分隐私账户，传入当前搜索的噪声乘数、采样概率，。。。和最大组合数
        accountant = Accountant(
            noise_multiplier=mid_noise_multiplier,
            sampling_probability=batch_size / dataset_size,
            delta=delta,
            eps_error=0.1,
            max_compositions=round(dataset_size / batch_size) * epoch,
        )
        # 计算给定组合下的eps值
        _, eps_estimate, _ = accountant.compute_epsilon(
            num_compositions=round(dataset_size / batch_size) * epoch)
        # print(eps_estimate)
        # 满足给定的eps值，退出查找
        if abs(eps_estimate - eps) < 0.0001:
            break
        # less noise
        if eps_estimate > eps:
            # 增加noise
            start_noise_multiplier = mid_noise_multiplier
        else:
            # 减小noise
            end_noise_multiplier = mid_noise_multiplier

    return mid_noise_multiplier


# noise_factor 控制添加噪声的强度
# norm_c: 最大范数裁剪的阈值,默认值为 1.0
def add_noise(embeddings, noise_factor=0.5, norm_c=1.0, add_noise=True, mean=0, std=1):
    if add_noise:
        # 最大范数裁剪
        embeddings = _max_norm_clip(embeddings, norm_c)
        # 生成与 embeddings 形状相同的随机噪声向量noise_embeds, 其中每个元素从均值为0、标准差为1的正态分布中采样。
        # 噪声向量会乘以noise_factor, 以控制噪声的强度。
        noise_embeds = noise_factor * torch.normal(mean=mean, std=std, size=embeddings.shape).to(embeddings.device,
                                                                                                 dtype=torch.bfloat16)
        embeddings = noise_embeds + embeddings
    return embeddings


def _max_norm_clip(embeddings, norm_c=1.0):
    shape = embeddings.shape
    # 把向量重塑为二维张量，第一维是批次大小
    embeddings = embeddings.reshape(shape[0], -1)
    # 计算L2范数
    total_norm = torch.norm(embeddings, dim=-1)
    # print(total_norm.mean())
    # 计算裁剪系数
    clip_coef = norm_c / (total_norm + 1e-6)
    # 限制不能超过1
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    embeddings = torch.multiply(embeddings, clip_coef_clamped.unsqueeze(-1))
    embeddings = embeddings.reshape(shape)

    return embeddings


def main():
    # Llama3-8B-Instruct l2-sensitivity of embedd_token:
    # Llama2-7B-chat l2-sensitivity of embedd_token: 1.8

    # model = LlamaForCausalLM.from_pretrained("../../Models/Llama-2-7b-chat-hf",
    #                                          device_map="auto", torch_dtype=torch.float32)
    # l1_sensitivity, l2_sensitivity = cal_sensitivity_llama_embedding(model)
    l2_sensitivity = cal_sensitivity_llama_embedding(model)
    print(f"emb: {l2_sensitivity:.2f}")
    eps = [1, 2, 4, 8, 16, 32, 64]
    delta = 0.001
    for i in eps:
        noise_b = matrix_gaussian_noise(epsilon=i, delta=delta, sensitivity=l2_sensitivity)
        print(f"aMVG noise_b is {noise_b:.2f}")


if __name__ == "__main__":
    # main()
    device = torch.device("cuda:1")
    # model = LlamaForCausalLM.from_pretrained("../../Models/Llama-2-7b-chat-hf")
    model = BertForSequenceClassification.from_pretrained("../../Models/bert-base-uncased")
    model = model.to(device)
    get_nearest_tokens(model)
