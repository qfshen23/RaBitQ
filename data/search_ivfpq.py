import numpy as np
import faiss
import os
from utils.io import read_vector_data  # 假设有这个自定义函数

def load_index_and_search(index_path, query_path, gt_path, nprobe=1000, k=10):
    # 从磁盘加载索引
    print(f"Loading index from {index_path}...")
    ivfpq_index = faiss.read_index(index_path)

    # 设置 nprobe
    ivfpq_index.nprobe = nprobe
    print(f"nprobe set to {nprobe}.")

    # 读取查询向量和 Ground Truth
    print("Reading query vectors...")
    query = read_vector_data(query_path).astype('float32')
    gt = read_vector_data(gt_path)

    # 查询索引
    print("Searching the index...")
    distances, indices = ivfpq_index.search(query, k)

    # 计算平均召回率
    def compute_average_recall(gt, indices, k):
        total_recall = 0
        nq = gt.shape[0]  # 查询数量

        for i in range(nq):
            recall = len(set(gt[i, :k]) & set(indices[i, :k])) / k
            total_recall += recall

        return total_recall / nq

    average_recall = compute_average_recall(gt, indices, k)
    print(f"Average Recall@{k}: {average_recall:.4f}")

if __name__ == '__main__':
    dataset_name = 'laion-10M'
    output_dir = f'./data/{dataset_name}'
    nlist = 4096
    m = 16
    bits = 8
    index_filename = f"{dataset_name}_nlist{nlist}_m{m}_bits{bits}.index"
    index_path = os.path.join(output_dir, index_filename)

    query_path = f'./data/{dataset_name}/query.10k.fbin'
    gt_path = f'./data/{dataset_name}/gt.10k.ibin'

    load_index_and_search(index_path, query_path, gt_path)
