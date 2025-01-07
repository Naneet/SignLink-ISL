def preload_data(dataloader, device) ->list:
    """
    Transfers batches of data from a DataLoader to the specified device.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader that provides batches of data.
        device (torch.device): The device (CPU or CUDA) to which the data batches will be transferred.

    Returns:
        list: A list of tuples, where each tuple contains:
            - X (torch.tensor): The input data (e.g., video frames, features) from the batch, transferred to the specified device.
            - y (torch.tensor): The corresponding labels from the batch, transferred to the specified device.

    Example:
        data = preload_data(train_dataloader, torch.device('cuda'))
    """
    data_tensor_list = []
    for X, y in dataloader:
        data_tensor_list.append((X.to(device),y.to(device)))
    return data_tensor_list
    