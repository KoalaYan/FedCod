import numpy as np

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
            M = np.append(M, zeros)
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
        return np.concatenate((I, P), axis=0)

    def multiply(self, M, G):
        R = np.matmul(G, M)
        return R[:, np.newaxis, :]

    def encode_RS(self, M, k, r):
        G = self.RS(k + r, k)
        if M.shape[0] % k != 0:
            zeros = np.zeros(k - M.shape[0] % k)
            M = np.append(M, zeros)
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
            array = np.append(array, zeros)
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
            array = np.append(array, zeros)
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
        blocks = np.stack(blocks, axis=0)
        index = np.random.randint(low, high, size=(r, n))
        encoded_blocks = np.dot(index, blocks)
        return encoded_blocks

    def decoding(self, encoded_blocks, coefficient_matrix):
        if type(encoded_blocks) == isinstance(encoded_blocks, list):
            encoded_blocks = np.vstack(encoded_blocks)
        inverse_matrix = np.linalg.inv(coefficient_matrix)
        decoded_blocks = np.dot(inverse_matrix, encoded_blocks) #[inverse_matrix[i] @ encoded_blocks for i in range(self.k)]
        return decoded_blocks