import numpy as np
import faiss
import struct
import os
from utils.io import *

source = './data/'

if __name__ == '__main__':
    dataset = 'laion-10M'

    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'base.10M.fbin')
    query_path = os.path.join(path, f'query.10k.fbin')
    gt_path = os.path.join(path, f'gt.10k.ibin')

    base = read_vector_data(data_path)
    query = read_vector_data(query_path)
    gt = read_vector_data(gt_path)

    # 确保数据类型为 float32
    base = base.astype('float32')
    query = query.astype('float32')

    # 参数设置
    d = base.shape[1]  # 向量维度
    nlist = 4096  # 倒排单元数量
    m = 16  # 子空间数量
    bits = 16  # 每个子向量的量化位数
    k = gt.shape[1]  # 每个查询的真实最近邻数量

    # 1. 创建 IVFPQ 索引
    coarse_quantizer = faiss.IndexFlatL2(d)  # 粗量化器
    ivfpq_index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, m, bits)

    # 2. 训练索引
    print("Training IVFPQ index...")
    ivfpq_index.train(base)

    # 3. 添加向量到索引
    print("Adding base vectors to the index...")
    ivfpq_index.add(base)
    print(f"Total vectors in the index: {ivfpq_index.ntotal}")

    # 4. 设置查询参数
    nprobe = 100  # 探测单元数量（越高召回率越好，查询速度越慢）
    ivfpq_index.nprobe = nprobe

    # 5. 执行查询
    print("Searching the index...")
    distances, indices = ivfpq_index.search(query, k)

    def compute_average_recall(gt, indices, k):
        """
        计算所有查询的平均召回率 (Recall@k)
        :param gt: Ground truth 索引数组，形状为 (nq, k)
        :param indices: 查询返回的预测索引数组，形状为 (nq, k)
        :param k: 最近邻数量
        :return: 平均召回率
        """
        total_recall = 0
        nq = gt.shape[0]  # 查询数量

        for i in range(nq):
            # 计算每个查询的召回率
            recall = len(set(gt[i, :k]) & set(indices[i, :k])) / k
            total_recall += recall

        # 平均召回率
        return total_recall / nq

    # 假设 gt 和 indices 已经生成
    average_recall = compute_average_recall(gt, indices, k)
    print(f"Average Recall@{k}: {average_recall:.4f}")
