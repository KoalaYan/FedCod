import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
import numpy as np
from utils.config_to_arg import argument

def HAR_dataloader():
    df_train=pd.read_csv('./data/HAR/train.csv')
    X=pd.DataFrame(df_train.drop(['Activity','subject'], axis=1))
    y=df_train.Activity.values.astype(object)
    group=df_train.subject.values.astype(object)

    encoder=preprocessing.LabelEncoder()
    encoder.fit(y)
    y_train_csv=encoder.transform(y)
    scaler=StandardScaler()
    X_train_csv=scaler.fit_transform(X)

    # X_train_csv, y_train_csv = torch.tensor(X_train_csv, dtype=torch.float), torch.tensor(y_train_csv, dtype=torch.float)
    
    temp_data = [[] for _ in range(30)]
    temp_label = [[] for _ in range(30)]
    for i in range(len(group)):
        g = group[i] - 1
        temp_data[g].append(X_train_csv[i])
        temp_label[g].append(y_train_csv[i])


    df_test=pd.read_csv('./data/HAR/test.csv')
    Xx=pd.DataFrame(df_test.drop(['Activity','subject'],axis=1))
    yy=df_test.Activity.values.astype(object)
    group_1=df_test.subject.values.astype(object)

    encoder_1=preprocessing.LabelEncoder()
    encoder_1.fit(yy)
    y_test_csv=encoder_1.transform(yy)
    scaler=StandardScaler()
    X_test_csv=scaler.fit_transform(Xx)

    # X_test_csv, y_test_csv = torch.tensor(X_test_csv, dtype=torch.float), torch.tensor(y_test_csv, dtype=torch.float)

    for i in range(len(group_1)):
        g = group_1[i] - 1
        temp_data[g].append(X_test_csv[i])
        temp_label[g].append(y_test_csv[i])

    each_worker_data = []
    each_worker_label = []
    server_data = []
    server_label = []
    X_test = []
    y_test = []

    for i in range(30):
        X_t, Xx_test, y_t, yy_test = train_test_split(temp_data[i],temp_label[i],test_size=0.2,random_state=100)
        X_test = X_test + Xx_test
        y_test = y_test + yy_test
        X_worker, X_server, y_worker, y_server = train_test_split(X_t,y_t,test_size=0.03,random_state=100)
        X_worker, y_worker = torch.tensor(X_worker, dtype=torch.float), torch.tensor(y_worker, dtype=torch.long)
        X_worker = X_worker.cpu()
        y_worker = y_worker.cpu()
        each_worker_data.append(X_worker)
        each_worker_label.append(y_worker)
        server_data = server_data + X_server
        server_label = server_label + y_server
    #     print(i, len(each_worker_data[i]), len(each_worker_label[i]))
    # print(len(server_data), len(server_label))
    # print(len(X_test), len(y_test))

    server_data, server_label = torch.tensor(server_data, dtype=torch.float), torch.tensor(server_label, dtype=torch.long)
    server_data = server_data.cpu()
    server_label = server_label.cpu()
    X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)
    return each_worker_data, each_worker_label, server_data, server_label, DataLoader(TensorDataset(X_test, y_test), len(X_test), drop_last = True, shuffle=False)

def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        train_data = DataLoader(torchvision.datasets.FashionMNIST(root = './data/', train=True, download=True, transform=transforms.ToTensor()), 60000, drop_last = True, shuffle=True)
        test_data = DataLoader(torchvision.datasets.FashionMNIST(root = './data/', train=False, download=True, transform=transforms.ToTensor()), 250, drop_last = True, shuffle=False)
    elif dataset == 'CIFAR10':        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trans1=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])
        trans2=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_data = DataLoader(torchvision.datasets.CIFAR10(root = './data/', train=True, download=True, transform=trans1), 50000, drop_last = True, shuffle=True)
        test_data = DataLoader(torchvision.datasets.CIFAR10(root = './data/', train=False, download=True, transform=trans2), 250, drop_last = True, shuffle=False)
    elif dataset == 'HAR':
        X_train, X_test, y_train, y_test = HAR_dataloader()
        X_train, X_test, y_train, y_test = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float), torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
        train_data = DataLoader(TensorDataset(X_train, y_train),6984, drop_last = True, shuffle=True)
        test_data = DataLoader(TensorDataset(X_test, y_test), 368, drop_last = True, shuffle=False)
    else:
        raise NotImplementedError
    return train_data, test_data


def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, dataset="FashionMNIST", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels
    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]      
    
    for _, item in enumerate(train_data):
        data, label = item
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.to(ctx).reshape(1,1,28,28)
            elif dataset == "CIFAR10":
                x = x.to(ctx).reshape(1,3,32,32)
            elif dataset == "HAR":
                x = x.to(ctx).reshape(1,561)
            else:
                raise NotImplementedError
            
            y = y.to(ctx).reshape(1)
            
            upper_bound = (y.cpu().numpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.cpu().numpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()
            
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.cpu().numpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.cpu().numpy()
            
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)
                    
    each_worker_data = [torch.cat(each_worker, dim=0).cpu() for each_worker in each_worker_data] 
    each_worker_label = [torch.cat(each_worker, dim=0).cpu() for each_worker in each_worker_label]
    
    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return each_worker_data, each_worker_label

def split_dataset(dataset, args):
    print(args.dataset)
    num_classes = 10
    if args.dataset == "FashionMNIST" or args.dataset == "CIFAR10":
        num_classes = 10
    elif args.dataset == "HAR":
        num_classes = 6

    ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, test_data = load_data(dataset)    
    # assign data to the server and clients
    each_worker_data, each_worker_label = assign_data(
                                            train_data, args.bias, ctx, num_labels=num_classes, 
                                            num_workers=args.num_users, dataset=args.dataset)
    
    for client_id in range(args.num_users):
        torch.save(each_worker_data[client_id], './distributed_dataset/data/'+str(client_id)+'.pt')
        torch.save(each_worker_label[client_id], './distributed_dataset/label/'+str(client_id)+'.pt')



if __name__ == '__main__':    
    args = argument()
    split_dataset('CIFAR10', args)