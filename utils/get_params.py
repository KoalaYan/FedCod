import torch
import numpy as np

def get_params(model_dict):
    param_1 = [model_dict[key] for key in model_dict.keys() if "num_batches_tracked" not in key]
    return param_1

def get_params_flatten(model_dict):
    # param_1 = []
    # for key in model_dict.keys():
    #     if "num_batches_tracked" not in key:
    #         item = model_dict[key].cpu().numpy().ravel()
    #         param_1.append(item)
    # param_1 = np.concatenate(param_1)
    
    param_1 = []
    for key in model_dict.keys():
        if "num_batches_tracked" not in key:
            item = model_dict[key]
            param_1.append(item.flatten())
    param_1 = torch.cat(param_1).cpu().numpy().astype(np.float64)
    return param_1
    
def get_updates(new_model_dict, model_dict):
    res = [(model_dict[i]-new_model_dict[i]) for i in range(len(new_model_dict))]
    return res

def get_updates_flatten(new_model_dict, model_dict):
    # res = np.array([])
    # for key in model_dict.keys():
    #     if "num_batches_tracked" not in key:
    #         item = model_dict[key]-new_model_dict[key]
    #         res = np.append(res, item.cpu().numpy().ravel())
    res = []
    for key in model_dict.keys():
        if "num_batches_tracked" not in key:
            item = model_dict[key] - new_model_dict[key]
            # 直接将计算结果添加到列表中，而不是使用 .cpu().numpy().ravel()
            res.append(item.flatten())
    # 将列表中的所有张量堆叠起来形成一个大的张量
    res = torch.cat(res).cpu().numpy().astype(np.float64)
    return res.ravel()

### WARNING!!! ONLY FOR NETWORK TEST!
def get_updates_flatten_network_test(new_model_dict, model_dict):
    res = np.array([])
    for key in model_dict.keys():
        if "num_batches_tracked" not in key:
            # item = (model_dict[key]-new_model_dict[key]).cpu().numpy().ravel()
            item = np.ones(shape=model_dict[key].cpu().numpy().ravel().shape())
            res = np.append(res, item)
        # res += np.random.random(res.size)
    return res

def rebuild_dict(params, model_dict):
    j = 0
    for name in model_dict.keys():
        if "num_batches_tracked" not in name:
            model_dict[name] = params[j]
            j += 1

    return model_dict

def rebuilt_dict_flatten(params, model_dict):
    data_list = []
    datalist = [model_dict[key].clone() for key in model_dict.keys() if "num_batches_tracked" not in key]
    idx = 0
    for data in datalist:
        size = 1
        for item in data.shape:
            size *= item

        temp = params[idx:(idx+size)].reshape(data.shape)
        data_list.append(temp)
        idx += size

    j = 0
    for name in model_dict.keys():
        if "num_batches_tracked" not in name:
            model_dict[name] = torch.tensor(data_list[j]).to(model_dict[name].device)
            j += 1

    return model_dict