import numpy as np
import faiss
import os
from utils.io import read_vector_data  # 假设有这个自定义函数

def build_and_save_index(data_path, dataset_name, output_dir, nlist=4096, m=16, bits=8):
    # 读取向量数据  
    print("Reading base vectors...")
    base = read_vector_data(data_path).astype('float32')
    d = base.shape[1]  # 向量维度

    # 构建粗量化器
    print("Creating coarse quantizer...")
    coarse_quantizer = faiss.IndexFlatL2(d)

    # 创建 IVFPQ 索引
    print("Creating IVFPQ index...")
    ivfpq_index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, m, bits)

    # 训练索引
    print("Training the index...")
    ivfpq_index.train(base)

    # 添加向量到索引
    print("Adding base vectors to the index...")
    ivfpq_index.add(base)
    print(f"Total vectors in the index: {ivfpq_index.ntotal}")

    # 动态生成索引文件名
    index_filename = f"{dataset_name}_nlist{nlist}_m{m}_bits{bits}.index"
    index_path = os.path.join(output_dir, index_filename)

    # 将索引存储到磁盘
    print(f"Saving index to {index_path}...")
    faiss.write_index(ivfpq_index, index_path)
    print("Index saved successfully.")

if __name__ == '__main__':
    dataset_name = 'laion-10M'
    dataset_path = f'./data/{dataset_name}/base.10M.fbin'
    output_dir = f'./data/{dataset_name}'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建并保存索引
    build_and_save_index(dataset_path, dataset_name, output_dir, nlist=4096, m=16, bits=8)
