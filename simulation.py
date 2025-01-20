import numpy as np

class Simulator:
    def __init__(self, n, k, r):
        self.n = n
        self.k = k
        self.r = r
        self.block_size = 435 / k
        
        self.download_time = []
        self.training_time = []
        self.upload_time = []

        mu_matrix = np.zeros(shape=(self.n+1,self.n+1))
        delta_matrix = np.zeros(shape=(self.n+1,self.n+1))
        for i in range(self.n+1):
            for j in range(self.n+1):
                # mu_matrix[i][j]  = 157.355 / 10
                mu_matrix[i][j]  = 200 / 8
                delta_matrix[i][j] = 23.11017404 / 10
                if (i == 0 and j == 1) or (i == 1 and j == 0):
                    mu_matrix[i][j]  = 1600 / 8
                    delta_matrix[i][j] = 23.11017404 

        self.mu_matrix = mu_matrix
        self.delta_matrix = delta_matrix

    def sample_bandwidth(self, i, j):

        mu = 157.355 / 10
        delta = 23.11017404 / 10
        # return self.block_size / np.random.normal(mu, delta)
        return self.block_size / np.random.normal(self.mu_matrix[i][j], self.delta_matrix[i][j])
    
    def sample_training_time(self, i):
        mu_train = 0.2
        delta_train = 0.05
        return np.random.normal(mu_train, delta_train)
    
    def sample_loadmodel_time(self):
        mu = 2
        delta = 0.5
        return np.random.normal(mu, delta)

    def clear(self):
        self.download_time = []
        self.training_time = []
        self.upload_time = []
    
    def train(self):
        for i in range(n):
            self.training_time.append(self.download_time[i]+self.sample_training_time(i))

    def download(self):
        return
    
    def upload(self):
        return
    

class BaselineSimulator(Simulator):
    def download(self):
        block_download_time = np.zeros((self.n,self.k+1))
        for i in range(self.n):
            for s in range(1,self.k+1):
                block_download_time[i][s] = block_download_time[i][s-1] + self.sample_bandwidth(0, i+1)

        for i in range(self.n):
            block_download_time[i] = np.sort(block_download_time[i])
            self.download_time.append(block_download_time[i][block_download_time[i] != 0][self.k-1])
    
    def upload(self):
        block_upload_time = np.zeros((self.n,self.k+1))

        for i in range(self.n):
            block_upload_time[i][0] = self.training_time[i]

        for i in range(self.n):
            for s in range(1,self.k+1):
                block_upload_time[i][s] = block_upload_time[i][s-1] + self.sample_bandwidth(0, i+1)

        for i in range(self.n):
            block_upload_time[i] = np.sort(block_upload_time[i])
            self.upload_time.append(block_upload_time[i][block_upload_time[i] != 0][self.k-1])

    def simulate(self):
        self.download()
        self.train()
        self.upload()
        print(self.download_time)
        print([self.upload_time[i] - self.training_time[i] for i in range(len(self.training_time))])
        return np.max(self.upload_time)
                      
class HitchhikerSimulator(Simulator):

    def sample_coding_time(self):
        mu_coding = 0.2
        delta_coding = 0.05
        return np.random.normal(mu_coding, delta_coding)

    def download(self):
        block_download_time = np.zeros((self.n,(self.n+1)*self.k+1))
        for i in range(1, self.n):
            block_download_time[i][0] = block_download_time[i-1][0] + self.sample_coding_time()

        for i in range(self.n):
            for s in range(1,self.k+1):
                block_download_time[i][s] = block_download_time[i][s-1] + self.sample_bandwidth(0, i+1)

        for i in range(self.n):
            for j in range(self.n):
                if  j == i:
                    continue

                for s in range(1,self.k+1):
                    block_download_time[j][s+(i+1)*self.k] = block_download_time[i][s] + self.sample_bandwidth(i+1, j+1) + self.sample_coding_time()

        for i in range(self.n):
            block_download_time[i] = np.sort(block_download_time[i])
            self.download_time.append(block_download_time[i][block_download_time[i] != 0][self.k-1])

    def upload(self):
        block_upload_time = np.zeros((self.n,self.n+self.k+self.r))

        for i in range(1, self.n):
            block_upload_time[i][0] = block_upload_time[i-1][0] + self.sample_coding_time()

        for i in range(self.n):
            for s in range(self.n):
                block_upload_time[i][s] = self.training_time[i]

        for i in range(self.n):
            for s in range(0,self.k+self.r):
                block_upload_time[i][s+self.n] = block_upload_time[i][s] + self.sample_bandwidth(i+1, int(i%self.n)+1) + self.sample_coding_time()

        self.upload_time = []
        for s in range(self.k+self.r):
            self.upload_time.append(np.sort(block_upload_time[:,self.n+s])[-1]+self.sample_bandwidth(0, int(s%self.n)+1))

        self.upload_time = np.sort(self.upload_time)

    def simulate(self):
        self.download()
        self.train()
        self.upload()
        print(self.download_time)
        print([self.upload_time[self.k-1] - self.training_time[i] for i in range(len(self.training_time))])
        return self.upload_time[self.k-1]




n = 3
k = 6
r = 6
res = []
baseline = BaselineSimulator(n, k, r)
for i in range(1):
    res.append(baseline.simulate())
    baseline.clear()
print('baseline', np.mean(res))

rr = []
for r in range(12):
    res = []
    hitchhiker = HitchhikerSimulator(n, k, r)
    for i in range(1):
        res.append(hitchhiker.simulate())
        hitchhiker.clear()
    rr.append(np.mean(res))
    print(r, np.mean(res))

print(np.argmin(rr), np.min(rr))

    