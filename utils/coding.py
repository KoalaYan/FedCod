import torch
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class Coding:
    def Cauchy(self, m, n):
        x = torch.arange(n + 1, n + m + 1, device=device)
        y = torch.arange(1, n + 1, device=device)
        x = x.view(-1, 1)
        diff_matrix = x - y
        cauchym = 1.0 / diff_matrix
        return cauchym

    def RS(self, n, k):
        I = torch.eye(k, device=device)
        P = self.Cauchy(n - k, k)
        return torch.cat((I, P), dim=0)

    def multiply(self, M, G):
        count = 0
        D = M[0].shape
        X = 1
        Y = D[-1]
        N, K = G.shape
        R = torch.zeros((N, X, Y), device=device)
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
            zeros = torch.zeros(k-M.shape[0] % k, device=device)
            M = torch.cat([M, zeros])
        M = M.view(-1)
        M = torch.split(M, M.shape[0] // k)
        return self.multiply(M, G)

    def decode_RS(self, M, k, r, index):
        G = self.RS(k + r, k)
        G = G[index]
        # print G
        return self.multiply(M, np.linalg.inv(G))


class OptimizedCoding:
    def Cauchy(self, m, n):
        x = torch.arange(n + 1, n + m + 1, device=device)
        y = torch.arange(1, n + 1, device=device)
        diff_matrix = x[:, None] - y
        cauchym = 1.0 / diff_matrix
        return cauchym

    def RS(self, n, k):
        I = torch.eye(k, device=device)
        P = self.Cauchy(n - k, k)
        return torch.cat((I, P), dim=0).to(torch.float)

    def multiply(self, M, G):
        R = torch.matmul(G, M)
        return R[:, None, :]

    def encode_RS(self, M, k, r):
        G = self.RS(k + r, k)
        if M.shape[0] % k != 0:
            zeros = torch.zeros(k - M.shape[0] % k, device=device)
            M = torch.cat([M, zeros])
        M = M.view(k, -1)
        # print(M.shape, G.shape)
        return self.multiply(M, G)

    def decode_RS(self, M, k, r, index):
        G = self.RS(k + r, k)
        G = G[index]
        I = torch.eye(G.shape[0], device=device)
        inv = torch.linalg.solve(G, I)
        return self.multiply(M, inv)
    
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
            zeros = torch.zeros(k-array.shape[0] % k, device=device)
            array = torch.cat([array, zeros])
        array = array.view(-1)
        array = torch.split(array, array.shape[0] // k)
        I = torch.eye(k, device=device)
        # time.sleep(10)
        array = [torch.cat([I[i],array[i]]) for i in range(k)]
        return array
    
    def multiply(self, M, G):
        R = torch.matmul(G, M)
        return R[:, None, :]

    def encoding(self, blocks, low, high, r):
        n = len(blocks)
        blocks = torch.vstack(blocks)
        index = torch.randint(low, high, size=(r, n), device=blocks.device).to(torch.float)
        # index = np.random.rand(r, n)

        encoded_blocks = torch.matmul(index, blocks)
        return encoded_blocks

    def decoding(self, encoded_blocks, coefficient_matrix):
        encoded_blocks = torch.vstack(encoded_blocks)
        inverse_matrix = torch.linalg.inv(coefficient_matrix)
        decoded_blocks = torch.matmul(inverse_matrix, encoded_blocks) #[inverse_matrix[i] @ encoded_blocks for i in range(self.k)]
        return decoded_blocks
    
def test_coding_time(test_mode, param_volume=6e7, download_k=9, upload_k=9, upload_r=3, repeat=10):

     # 60 million parameters for ResNet152
    model_params = torch.rand(int(param_volume), device=device)  # Simulating model parameters
    print("Model parameters device:", model_params.device)

    if test_mode in ['all', 'download']:
        # NetworkCoding test
        nc = NetworkCoding(download_k)
        encode_time_list = []
        encoded_blocks = []
        blocks = nc.split(model_params)
        for t in range(max(download_k, repeat)):
            start_time = time.time()
            new_blocks = nc.encoding(blocks, 0, 128, 1)
            end_time = time.time()
            encode_time_list.append(end_time - start_time)
            if t < download_k:
                encoded_blocks.append(new_blocks[0])

        print(f"Encoding time: {np.average(encode_time_list)} seconds")

        decode_time_list = []
        for t in range(repeat):
            sample_index = torch.randperm(len(encoded_blocks))[:download_k]
            indexs = [encoded_blocks[i][:download_k] for i in sample_index]
            idx_list = torch.vstack(indexs)
            selected_blocks = [encoded_blocks[i][download_k:] for i in sample_index]
            start_time = time.time()
            decoded_blocks = nc.decoding(selected_blocks, idx_list)
            end_time = time.time()
            decode_time_list.append(end_time - start_time)
        print(f"Decoding time: {np.average(decode_time_list)} seconds")

        # 清理所有cuda memory占用
        del nc, encode_time_list, encoded_blocks, blocks, new_blocks, sample_index, indexs, idx_list, selected_blocks, decoded_blocks, decode_time_list
        torch.cuda.empty_cache()

    if test_mode in ['all', 'upload']:
        # Coded Aggregation test
        coding = OptimizedCoding()

        encode_time_list = []
        model_local = []
        for t in range(repeat):
            start_time = time.time()
            model_local = coding.encode_RS(model_params, upload_k, upload_r)
            end_time = time.time()
            encode_time_list.append(end_time - start_time)
            if t != repeat - 1:
                # release memory for the next iteration
                model_local = []
                torch.cuda.empty_cache()
        print(f"Coded Aggregation encoding time: {np.average(encode_time_list)} seconds")

        decode_time_list = []
        for t in range(repeat):
            sample_index = torch.randperm(upload_k + upload_r)[:upload_k]
            selected_blocks = torch.cat([model_local[i] for i in sample_index], dim=0)
            start_time = time.time()
            model_glob = coding.decode_RS(selected_blocks, upload_k, upload_r, sample_index)
            end_time = time.time()
            decode_time_list.append(end_time - start_time)
        print(f"Coded Aggregation decoding time: {np.average(decode_time_list)} seconds")
        del coding, encode_time_list, model_local, sample_index, selected_blocks, model_glob, decode_time_list
        torch.cuda.empty_cache()


# print("Using device:", device)
# # Test for different parameter volumes
# for param_volume in [6e7, 1e8, 5e8, 1e9]: # 
#     print(f"Testing with {param_volume} parameters:")
#     test_coding_time('all', param_volume=param_volume, repeat=10)
#     torch.cuda.empty_cache()
#     print("\n" + "="* 50 + "\n")

# # Test for different download_k
# for download_k in [10, 50, 100, 200]:
#     print(f"Testing with download_k {download_k}:")
#     test_coding_time('download', param_volume=6e7, download_k=download_k)
#     print("\n" + "="* 50 + "\n")

# # Test for different upload_k and upload_r
# for upload_k in [10, 50, 100, 200]:
#     upload_r = int(upload_k * 0.5)  # Example: 50% of upload_k
#     print(f"Testing with upload_k {upload_k} and upload_r {int(upload_r)}:")
#     test_coding_time('upload', upload_k=upload_k, upload_r=upload_r)
#     print("\n" + "="* 50 + "\n")