def preload_data(dataloader,device):
    data_tensor_list = []
    for X, y in dataloader:
        data_tensor_list.append((X.to(device),y.to(device)))
    return data_tensor_list
    