import torch
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

import models
from copy import deepcopy
import utils

def cal_modelDict(global_update, lr, net, BN):
    model_dict = deepcopy(net.state_dict())

    datalist = []
    if BN:
        for data in model_dict.values():
            datalist.append(data.clone())
    else:
        for name, _ in net.named_parameters():
            datalist.append(model_dict[name].clone())

    data_list = []
    idx = 0
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item 
        temp = data - lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size

    if BN:
        for i, name in enumerate(model_dict.keys()):
            model_dict[name] = data_list[i]
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]
    return model_dict


def fedavg(gradients, net, args):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    n = len(gradients)
    # print(gradients)
    # print(gradients[0][0].shape, type(gradients[0][0]))
    
    new_param_list = []

    weights = 1.0 / n

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        # new_param_list.append([x * weights for x in gradients[i]])
        new_param_list.append(gradients[i] * weights)
    
    # update the global model
    global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)
    # print('update:', torch.norm(global_update))

    idx = 0
    model_dict = net.state_dict()
    datalist = []
    datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]

    data_list = []
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item
            
        temp = data - global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size
    
    j = 0
    for name in model_dict.keys():
        if "num_batches_tracked" not in name:
            model_dict[name] = data_list[j]
            j += 1

    return model_dict

def fltrust(param_list, net, ctx, args):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """

    n = len(param_list) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = torch.squeeze(param_list[-1])
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = torch.squeeze(each_param_list)

        cos_sim_temp = torch.dot(baseline, each_param_array) / (torch.norm(baseline) + 1e-9) / (torch.norm(each_param_array) + 1e-9)

        cos_sim.append(cos_sim_temp)

    # print(cos_sim)
    cos_sim = torch.stack(cos_sim)[:-1]

    zero = torch.tensor(np.zeros(cos_sim.shape)).to(ctx)
    cos_sim = torch.maximum(cos_sim, zero) # relu

    normalized_weights = cos_sim / (torch.sum(cos_sim) + 1e-9) # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (torch.norm(param_list[i]) + 1e-9) * torch.norm(baseline))
    
    # update the global model
    global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    idx = 0
    model_dict = deepcopy(net.state_dict())

    datalist = []    
    if args.BN:
        datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    else:
        datalist = [model_dict[name].clone() for name, _ in net.named_parameters()]

    data_list = []
    
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item

        temp = data - args.global_lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size

    if args.BN:
        j = 0
        for name in model_dict.keys():
            if "num_batches_tracked" not in name:
                model_dict[name] = data_list[j]
                j += 1
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]

    return model_dict

def fedGrad(grad_list, niter, net, args, log_file):
    model_dict = deepcopy(net.state_dict())
        
    datalist = []
    if args.BN:
        datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    else:
        datalist = [model_dict[name].clone() for name, _ in net.named_parameters()]

    param_list = []
            
    for grad in grad_list:
        data_item = []
        idx = 0
        for data in datalist:
            size = 1
            for item in data.shape:
                size *= item
            temp = data - args.global_lr * grad[idx:(idx+size)].reshape(data.shape)
            data_item.append(temp)
        param_list.append(data_item)

    x = []
    for grad in grad_list:
        grd = np.array(grad.cpu())
        x.append(grd.reshape(-1))

    X = (x - np.mean(x))/(np.std(x))


    res = classify.Classify_kmeans(X)

    num = 0
    right = 0.0
    false_pos = 0.0
    false_neg = 0.0
    # print(res)
    for i in range(len(res)):
        if res[i] == 0:
            if i < args.nbyz:
                right += 1
            else:
                false_pos += 1
        else:
            if i < args.nbyz:
                false_neg += 1
            num += 1

    n = len(grad_list)
    if args.byz_type != "no":
        print("Predict accuracy rate: ", right/args.nbyz)
        if niter % 10 == 0:
            utils.write_log(log_file, "Iteration %02d. Detect Acc %0.4f" % (niter, right / args.nbyz) + '\n')
            utils.write_log(log_file, "Iteration %02d. False Positive Rate %0.4f" % (niter, false_pos / (n-args.nbyz)) + '\n')
            utils.write_log(log_file, "Iteration %02d. False Negative Rate %0.4f" % (niter, false_neg / args.nbyz) + '\n')


    if niter % 10 == 0:
        utils.write_log(log_file, "Iteration %02d. Detect Acc %0.4f" % (niter, right / args.nbyz) + '\n')
        # HERE
        if args.dataset == "HAR":
            id_flag = '/'
        else:
            if args.bias == 0.1:
                id_flag = "/iid/"
            else:
                id_flag = "/noniid/"

        path = "./grad_pca/"+ args.dataset + id_flag + args.byz_type + "/" + args.aggregation + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        # pca = PCA(n_components=2)
        # newX = pca.fit_transform(X)
        # np.save(path+str(niter)+'.npy', newX)

        # OVER

    n = len(grad_list)
    new_param_list = []
    weights = 1 / num 

    for i in range(n):
        if res[i] == 1:
            new_param_list.append(grad_list[i] * weights / (torch.norm(grad_list[i]) + 1e-9))
    
    global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    idx = 0
    
    data_list = []
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item
            
        temp = data - args.global_lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size
    
    if args.BN:
        j = 0
        for name in model_dict.keys():
            if "num_batches_tracked" not in name:
                model_dict[name] = data_list[j]
                j += 1
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]

    return model_dict

def trim(gradients, net, args):
    v_tran = torch.cat(gradients, dim=1)
    n_attackers = args.nbyz
    sorted_updates = torch.sort(v_tran, 1)[0]
    global_update = torch.mean(sorted_updates[:,n_attackers:-n_attackers], 1) if n_attackers else torch.mean(sorted_updates,1)

    idx = 0
    model_dict = net.state_dict()
    datalist = []
    if args.BN:
        datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    else:
        datalist = [model_dict[name].clone() for name, _ in net.named_parameters()]

    data_list = []
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item
            
        temp = data - args.global_lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size
    
    if args.BN:
        j = 0
        for name in model_dict.keys():
            if "num_batches_tracked" not in name:
                model_dict[name] = data_list[j]
                j += 1
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]

    return model_dict


def krum(gradients, net, args):
    n_attackers = args.nbyz
    v_tran = torch.cat(gradients, dim=1)
    nusers = v_tran.shape[1]
    candidates = []
    candidate_indices = []
    remaining_updates = v_tran
    all_indices = np.arange(nusers)

    distances = []
    for idx in range(remaining_updates.shape[1]):
        update = remaining_updates[:, idx].reshape((remaining_updates[:, idx].shape[0], 1))
        distance = torch.norm((remaining_updates - update), dim=0) ** 2        
        distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)

    distances = torch.sort(distances, dim=0)[0]
    scores = torch.sum(distances[:remaining_updates.shape[1]-2-n_attackers,:], dim=0)
    indices = torch.argsort(scores)

    candidate_indices.append(all_indices[indices[0].cpu().numpy()])
    all_indices = np.delete(all_indices, indices[0].cpu().numpy())
    candidates = remaining_updates[:,indices[0]].reshape((remaining_updates[:,indices[0]].shape[0], 1)) if not len(candidates) else torch.cat((candidates, remaining_updates[:,indices[0]]), dim=1)
    remaining_updates = remaining_updates[:,:indices[0]]


    global_update = torch.mean(candidates, dim=1)
    
    idx = 0
    model_dict = net.state_dict()
    datalist = []
    if args.BN:
        datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    else:
        datalist = [model_dict[name].clone() for name, _ in net.named_parameters()]

    data_list = []
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item
            
        temp = data - args.global_lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size
    
    if args.BN:
        j = 0
        for name in model_dict.keys():
            if "num_batches_tracked" not in name:
                model_dict[name] = data_list[j]
                j += 1
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]

    return model_dict

def multikrum(gradients, net, args):
    n_attackers = args.nbyz
    v_tran = torch.cat(gradients, dim=1)
    nusers = v_tran.shape[1]
    candidates = []
    candidate_indices = []
    remaining_updates = v_tran
    all_indices = np.arange(nusers)

    while remaining_updates.shape[1] > 2*n_attackers+2:
        distances = []
        for idx in range(remaining_updates.shape[1]):
            update = remaining_updates[:, idx].reshape((remaining_updates[:, idx].shape[0], 1))
            distance = torch.norm((remaining_updates - update), dim=0) ** 2
            distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)
            
        distances = torch.sort(distances, dim=0)[0]
        scores = torch.sum(distances[:remaining_updates.shape[1]-2-n_attackers,:], dim=0)
        indices = torch.argsort(scores)

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[:,indices[0]].reshape((remaining_updates[:,indices[0]].shape[0], 1)) if not len(candidates) else torch.cat((candidates, remaining_updates[:,indices[0]].reshape(-1,1)), dim=1)
        # remaining_updates = remaining_updates[:,:indices[0]]
        remaining_updates = torch.cat((remaining_updates[:,:indices[0]], remaining_updates[:,indices[0]+1:]), 1)

    global_update = torch.mean(candidates, dim=1)
    
    idx = 0
    model_dict = net.state_dict()
    datalist = []
    if args.BN:
        datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    else:
        datalist = [model_dict[name].clone() for name, _ in net.named_parameters()]

    data_list = []
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item
            
        temp = data - args.global_lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size
    
    if args.BN:
        j = 0
        for name in model_dict.keys():
            if "num_batches_tracked" not in name:
                model_dict[name] = data_list[j]
                j += 1
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]

    return model_dict

def bulyan(gradients, net, args):
    n_attackers = args.nbyz
    v_tran = torch.cat(gradients, dim=1)
    nusers = v_tran.shape[1]
    candidates = []
    candidate_indices = []
    remaining_updates = v_tran
    all_indices = np.arange(nusers)

    while remaining_updates.shape[1] > 2*n_attackers+2:
        distances = []
        for idx in range(remaining_updates.shape[1]):
            update = remaining_updates[:, idx].reshape((remaining_updates[:, idx].shape[0], 1))
            distance = torch.norm((remaining_updates - update), dim=0) ** 2
            distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)
            
        distances = torch.sort(distances, dim=0)[0]
        scores = torch.sum(distances[:remaining_updates.shape[1]-2-n_attackers,:], dim=0)
        indices = torch.argsort(scores)

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[:,indices[0]].reshape((remaining_updates[:,indices[0]].shape[0], 1)) if not len(candidates) else torch.cat((candidates, remaining_updates[:,indices[0]].reshape(-1,1)), dim=1)
        # remaining_updates = remaining_updates[:,:indices[0]]
        remaining_updates = torch.cat((remaining_updates[:,:indices[0]], remaining_updates[:,indices[0]+1:]), 1)

    # mid_global_update = torch.mean(candidates, dim=1)
    sorted_updates = torch.sort(candidates, 1)[0]
    global_update = torch.mean(sorted_updates[:,n_attackers:-n_attackers], 1) if n_attackers else torch.mean(sorted_updates,1)

    idx = 0
    model_dict = net.state_dict()
    datalist = []
    if args.BN:
        datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    else:
        datalist = [model_dict[name].clone() for name, _ in net.named_parameters()]

    data_list = []
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item
            
        temp = data - args.global_lr * global_update[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size
    
    if args.BN:
        j = 0
        for name in model_dict.keys():
            if "num_batches_tracked" not in name:
                model_dict[name] = data_list[j]
                j += 1
    else:
        for i, (name, _) in enumerate(net.named_parameters()):
            model_dict[name] = data_list[i]

    return model_dict

