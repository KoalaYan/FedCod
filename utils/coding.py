import numpy as np
import time

class Coding:
    def Cauchy(self, m, n):
        x = np.array(range(n + 1, n + m + 1))
        y = np.array(range(1, n + 1))
        x = x.reshape((-1, 1))
        diff_matrix = x - y
        cauchym = 1.0 / diff_matrix
        return cauchym

    def RS(self, n, k):
        I = np.identity(k)
        P = self.Cauchy(n - k, k)
        return np.concatenate((I, P), axis=0)

    def multiply(self, M, G):
        count = 0
        D = M[0].shape
        X = 1
        Y = D[-1]
        N, K = G.shape
        R = np.zeros((N, X, Y))
        for i in range(N):
            # print G[i]
            for j in range(K):
                if G[i, j] != 0:
                    R[i] = R[i] + G[i, j] * M[j]
                    count += 1
        # logging.info(" matrix multiplication: %d" % (count,))
        return R

    def encode_RS(self, M, k, r):
        G = self.RS(k + r, k)
        # M = np.array_split(M, k)
        if M.shape[0] % k != 0:
            zeros = np.zeros(k-M.shape[0] % k)
            M = np.concatenate([M, zeros])
        M = M.reshape(-1)
        M = np.array_split(M, k)
        return self.multiply(M, G)

    def decode_RS(self, M, k, r, index):
        G = self.RS(k + r, k)
        G = G[index]
        # print G
        return self.multiply(M, np.linalg.inv(G))


class OptimizedCoding:
    def Cauchy(self, m, n):
        x = np.arange(n + 1, n + m + 1)
        y = np.arange(1, n + 1)
        diff_matrix = x[:, np.newaxis] - y
        cauchym = 1.0 / diff_matrix
        return cauchym

    def RS(self, n, k):
        I = np.identity(k)
        P = self.Cauchy(n - k, k)
        return np.concatenate((I, P), axis=0).astype(np.float64)

    def multiply(self, M, G):
        R = np.matmul(G, M)
        return R[:, np.newaxis, :]

    def encode_RS(self, M, k, r):
        G = self.RS(k + r, k)
        if M.shape[0] % k != 0:
            zeros = np.zeros(k - M.shape[0] % k)
            M = np.concatenate([M, zeros])
        M = M.reshape(k, -1)
        # print(M.shape, G.shape)
        return self.multiply(M, G)

    def decode_RS(self, M, k, r, index):
        G = self.RS(k + r, k)
        G = G[index]
        I = np.eye(G.shape[0])
        inv = np.linalg.solve(G, I)
        return self.multiply(M, inv)
    

class Ratelesscoding:
    def split(self, array, k):  # split the original matrix into k parts
        if array.shape[0] % k != 0:
            zeros = np.zeros(k-array.shape[0] % k)
            array = np.concatenate([array, zeros])
        array = array.reshape(-1)
        array = np.split(array, k)
        return array

    def generate(self, range, k, m):  # generate the coefficients
        index = np.random.randint(range, size=(k*m))
        index = index.reshape(-1, k)
        return index

    def encode(self, index, array):
        array = np.dot(index, array)
        array = np.concatenate((index, array), axis=1)
        return array

    def decode(self, index, array, k):
        array = array.reshape(k, -1)
        index = index.reshape(k, -1)
        array = np.linalg.solve(index, array)
        array = array.reshape(-1)
        return array


def structure(args):
    part_idx_list = [None] * args.num_users
    part_list = [None] * args.num_users
    for i in range(args.num_users):
        part_idx_list[i] = []
        part_list[i] = [None] * (args.upload_k + args.upload_r)

    return part_idx_list, part_list


class NetworkCoding:
    def __init__(self, k):
        self.k = k

    def split(self, array):  # split the original matrix into k parts
        k = self.k
        if array.shape[0] % k != 0:
            zeros = np.zeros(k-array.shape[0] % k)
            array = np.concatenate([array, zeros])
        array = array.reshape(-1)
        array = np.split(array, k)
        I = np.identity(k)
        array = [np.concatenate([I[i],array[i]]) for i in range(k)]
        return array
    
    def multiply(self, M, G):
        R = np.matmul(G, M)
        return R[:, np.newaxis, :]
    
    def encoding(self, blocks, low, high, r):
        n = len(blocks)
        # blocks = np.stack(blocks, axis=0)
        index = np.random.randint(low, high, size=(r, n))
        # index = np.random.rand(r, n)

        encoded_blocks = np.dot(index, blocks)
        return encoded_blocks

    def decoding(self, encoded_blocks, coefficient_matrix):
        if type(encoded_blocks) == isinstance(encoded_blocks, list):
            encoded_blocks = np.vstack(encoded_blocks)
        inverse_matrix = np.linalg.inv(coefficient_matrix.astype(np.int32))
        decoded_blocks = np.dot(inverse_matrix, encoded_blocks) #[inverse_matrix[i] @ encoded_blocks for i in range(self.k)]
        return decoded_blocks
    
def test_coding_time(param_volume=6e7, num_users=9, download_k=9, upload_k=9, upload_r=3):

     # 60 million parameters for ResNet152
    model_params = np.random.rand(int(param_volume))  # Simulating model parameters

    # NetworkCoding test
    nc = NetworkCoding(download_k)

    encode_time_list = []
    encoded_blocks = []
    for t in range(10):
        start_time = time.time()
        blocks = nc.split(model_params)
        blocks = nc.encoding(blocks, 0, 128, num_users)
        end_time = time.time()
        for block in blocks:
            encoded_blocks.append(block)
        encode_time_list.append(end_time - start_time)
    print(f"Encoding time: {np.average(encode_time_list)} seconds")

    decode_time_list = []
    for t in range(10):
        sample_index = np.random.choice(download_k*10, size=download_k, replace=False)
        indexs = [encoded_blocks[i][:download_k] for i in sample_index]
        idx_list = np.stack(indexs)
        rank_matrix = np.linalg.matrix_rank(idx_list)
        selected_blocks = [encoded_blocks[i][download_k:] for i in sample_index]
        start_time = time.time()
        decoded_blocks = nc.decoding(selected_blocks, idx_list)
        end_time = time.time()
        decode_time_list.append(end_time - start_time)
    print(f"Decoding time: {np.average(decode_time_list)} seconds")

    # Coded Aggregation test
    coding = OptimizedCoding()

    encode_time_list = []
    model_local = []
    for t in range(10):
        start_time = time.time()
        model_local = coding.encode_RS(model_params, upload_k, upload_r)
        end_time = time.time()
        encode_time_list.append(end_time - start_time)
    print(f"Coded Aggregation encoding time: {np.average(encode_time_list)} seconds")

    decode_time_list = []
    for t in range(10):
        sample_index = np.random.choice(upload_k+upload_r, size=upload_k, replace=False)
        selected_blocks = np.concatenate([model_local[i] for i in sample_index], axis=0)
        start_time = time.time()
        model_glob = coding.decode_RS(selected_blocks, upload_k, upload_r, sample_index)
        end_time = time.time()
        decode_time_list.append(end_time - start_time)
    print(f"Coded Aggregation decoding time: {np.average(decode_time_list)} seconds")

for param_volume in [6e7, 1e8, 2e8]:
    print(f"Testing with {param_volume} parameters:")
    test_coding_time(param_volume=param_volume, num_users=9, download_k=9, upload_k=9, upload_r=3)
    print("\n" + "="* 50 + "\n")

