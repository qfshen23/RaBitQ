import numpy as np
import faiss
import os
from utils.io import read_vector_data  # 假设有这个自定义函数

def build_and_save_index_with_gpu(data_path, dataset_name, output_dir, nlist=4096, m=16, bits=8, gpu_id=0):
    # 读取向量数据
    print("Reading base vectors...")
    base = read_vector_data(data_path).astype('float32')
    d = base.shape[1]  # 向量维度

    # 创建 GPU 资源管理对象
    print(f"Initializing GPU resources on GPU {gpu_id}...")
    gpu_resources = faiss.StandardGpuResources()

    # 创建粗量化器
    print("Creating coarse quantizer...")
    coarse_quantizer = faiss.IndexFlatL2(d)

    # 创建 IVFPQ 索引（在 CPU 上初始化）
    print("Creating IVFPQ index...")
    ivfpq_index_cpu = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, m, bits)

    # 将索引移动到 GPU 上
    print(f"Transferring index to GPU {gpu_id}...")
    ivfpq_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, ivfpq_index_cpu)

    # 训练索引（在 GPU 上进行）
    print("Training the index on GPU...")
    ivfpq_index.train(base)

    # 添加向量到索引（在 GPU 上进行）
    print("Adding base vectors to the index on GPU...")
    ivfpq_index.add(base)
    print(f"Total vectors in the index: {ivfpq_index.ntotal}")

    # 将索引移回 CPU 以保存到磁盘（FAISS 仅支持在 CPU 上保存索引）
    print("Transferring index back to CPU for saving...")
    ivfpq_index_cpu = faiss.index_gpu_to_cpu(ivfpq_index)

    # 动态生成索引文件名
    index_filename = f"{dataset_name}_nlist{nlist}_m{m}_bits{bits}_gpu.index"
    index_path = os.path.join(output_dir, index_filename)

    # 将索引存储到磁盘
    print(f"Saving index to {index_path}...")
    faiss.write_index(ivfpq_index_cpu, index_path)
    print("Index saved successfully.")

if __name__ == '__main__':
    dataset_name = 'laion-10M'
    dataset_path = f'./data/{dataset_name}/base.10M.fbin'
    output_dir = f'./data/{dataset_name}'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建并保存索引（使用 GPU）
    build_and_save_index_with_gpu(dataset_path, dataset_name, output_dir, nlist=4096, m=16, bits=8, gpu_id=0)