import numpy as np
from coding import NetworkCoding
from coding import OptimizedCoding

def test_network_coding_full_cycle():
    # 初始化NetworkCoding类
    k = 9  # 分成4块
    n = 12  # 编码成6块
    nc = NetworkCoding(k)
    
    # 生成一个随机数组
    original_array = np.random.randint(0, 100, size=(100))  # 假设每个块的大小为2x2
    
    # 使用split方法分割数组成k块
    blocks = nc.split(original_array)
    
    encoded_blocks = nc.encoding(blocks, 0, k, n)
    
    # 从n块中随机挑选k块进行解码
    random_indices = np.random.choice(n, k, replace=False)  # 随机选择不重复的k个索引
    
    idx_matrix = None
    part_list = []
    for i in random_indices:
        print(i)
        block = encoded_blocks[i]
        index = block[:k]
        data_block = block[k:]

        if idx_matrix is not None:
            idx_matrix = np.vstack((idx_matrix, index))
        else:
            idx_matrix = index
        part_list.append(data_block)

    decoded_blocks = nc.decoding(part_list.copy(), idx_matrix.copy())
    
    # 将解码后的块重新组合成一个数组，并与原始数组比较差异
    decoded_array = np.hstack([block for block in decoded_blocks])
    difference = original_array - decoded_array
    
    print("Original Array:")
    print(original_array)
    print("\nDecoded Array:")
    print(decoded_array)
    print("\nDifference:")
    print(difference)
    print("\nDifference Norm:")
    print(np.linalg.norm(difference))

def test_optimized_coding_full_cycle():
    # 初始化OptimizedCoding类
    k = 9  # 分成4块
    r = 9  # 冗余块数
    n = 18  # 编码成6块
    oc = OptimizedCoding()
    
    # 生成一个随机数组
    original_array = np.random.randint(0, 100, size=(100))  # 假设每个块的大小为2x2
    
    # 使用split方法分割数组成k块

    encoded_blocks = oc.encode_RS(original_array, k, r)
    
    # 从n块中随机挑选k块进行解码
    random_indices = np.random.choice(n, k, replace=False)  # 随机选择不重复的k个索引
    
    model_local = np.array([])
    for i in random_indices:
        print(i)
        block = encoded_blocks[i]
        model_local = np.append(model_local, block)
    
    model_local = model_local.reshape(k, -1)
    decoded_blocks = oc.decode_RS(model_local, k, r, random_indices)

    decoded_array = np.reshape(decoded_blocks, -1)[:len(original_array)]
    difference = original_array - decoded_array
    
    print("Original Array:")
    print(original_array)
    print("\nDecoded Array:")
    print(decoded_array)
    print("\nDifference:")
    print(difference)
    print("\nDifference Norm:")
    print(np.linalg.norm(difference))



def test_optimized_coding_add():
    # 初始化OptimizedCoding类
    k = 9  # 分成4块
    r = 9  # 冗余块数
    n = 18  # 编码成6块
    oc = OptimizedCoding()

    model_size = 1000
    
    # 生成一个随机数组
    original_array = np.random.rand(9,model_size)  # 假设每个块的大小为2x2
    
    random_indices = np.random.choice(n, k, replace=False)  # 随机选择不重复的k个索引
    model_local = np.array([])
    # 使用split方法分割数组成k块
    for i in range(9):
        encoded_blocks = oc.encode_RS(original_array[i].reshape(-1), k, r)
        print(i, encoded_blocks.shape)
    
    # 从n块中随机挑选k块进行解码
        if i == 0:
            for j in random_indices:
                block = encoded_blocks[j]
                model_local = np.append(model_local, block)
            model_local = model_local.reshape(k, -1)
            print(model_local.shape)
        else:
            for t, j in enumerate(random_indices):
                block = encoded_blocks[j]
                model_local[t] = model_local[t] + block.reshape(-1)
            
    decoded_blocks = oc.decode_RS(model_local, k, r, random_indices)

    decoded_array = np.reshape(decoded_blocks, -1)[:model_size]
    difference = np.sum(original_array, axis=0).reshape(-1) - decoded_array
    
    print("Original Array:")
    print(np.sum(original_array, axis=0))
    print("\nDecoded Array:")
    print(decoded_array)
    print("\nDifference:")
    print(difference)
    print("\nDifference Norm:")
    print(np.linalg.norm(difference))
# 运行测试
# test_network_coding_full_cycle()
test_optimized_coding_add()