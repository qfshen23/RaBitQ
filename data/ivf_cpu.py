import numpy as np
import time
import faiss
import os
from utils.io import *

# 构建索引并存储到磁盘
def build_and_save_ivf_index(data_path, index_path, nlist=100, d=128):
    """
    构建 IVF 索引并保存到磁盘
    :param data_path: 基础数据路径
    :param index_path: 索引保存路径
    :param nlist: 倒排列表的数量
    :param d: 向量维度
    """
    print("Reading base vectors...")
    base_vectors = read_vector_data(data_path).astype('float32')  # 假设数据存储为 npy 格式
    print(f"Base vectors shape: {base_vectors.shape}")

    # 构建粗量化器
    print("Creating coarse quantizer...")
    quantizer = faiss.IndexFlatL2(d)  # 基于 L2 距离的粗量化器

    # 构建 IVF 索引
    print(f"Creating IVF index with nlist={nlist}...")
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # 训练索引
    print("Training the index...")
    index.train(base_vectors)

    # 添加向量到索引
    print("Adding vectors to the index...")
    index.add(base_vectors)
    print(f"Total vectors in index: {index.ntotal}")

    # 保存索引到磁盘
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    print("Index saved successfully.")

def load_and_search_index(index_path, query_path, k=10, nprobe=10):
    """
    从磁盘加载 IVF 索引并进行搜索，同时计算 QPS。
    """
    print(f"Loading index from {index_path}...")
    index = faiss.read_index(index_path)

    # 设置 nprobe
    index.nprobe = nprobe
    print(f"nprobe set to {nprobe}.")

    # 读取查询向量
    print("Reading query vectors...")
    query_vectors = read_vector_data(query_path).astype('float32')  # 假设查询数据存储为 npy 格式
    print(f"Query vectors shape: {query_vectors.shape}")

    # 搜索索引并计时
    print("Searching the index...")
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    end_time = time.time()

    # 计算 QPS
    elapsed_time = end_time - start_time
    num_queries = query_vectors.shape[0]
    qps = num_queries / elapsed_time

    # 输出搜索结果和 QPS
    print("Search completed.")
    print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Queries Per Second (QPS): {qps:.2f}")

    # 计算平均召回率
    print("Calculating average recall...")
    gt = read_vector_data(gt_path).astype('int32')

    def compute_average_recall(gt, indices, k):
        total_recall = 0
        nq = gt.shape[0]  # 查询数量

        for i in range(nq):
            recall = len(set(gt[i, :k]) & set(indices[i, :k])) / k
            total_recall += recall

        return total_recall / nq

    average_recall = compute_average_recall(gt, indices, k)
    print(f"Average Recall@{k}: {average_recall:.4f}")

    return distances, indices, qps


# 主函数
if __name__ == '__main__':
    dataset_name = 'laion-10M'
    dataset_path = f'./data/{dataset_name}/base.10M.fbin'
    output_dir = f'./data/{dataset_name}'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    query_path = f'./data/{dataset_name}/query.10k.fbin'
    gt_path = f'./data/{dataset_name}/gt.10k.ibin'

    # 向量维度和倒排列表数量
    vector_dim = 512
    nlist = 4096

    index_filename = f"{dataset_name}_ivf_nlist{nlist}.index"
    index_path = os.path.join(output_dir, index_filename)

    # 构建并保存索引
    # build_and_save_ivf_index(dataset_path, index_path, nlist=nlist, d=vector_dim)

    # 从磁盘加载索引并进行搜索
    load_and_search_index(index_path, query_path, k=1, nprobe=300)
  