import torch
import os

def pickle_data_path_lists(path : str, dataloader):
    """
    Save data from a dataloader to disk as separate tensor files.

    Args:
        path (str): Directory where the tensor files will be saved.
        dataloader (iterable): An iterable that yields tuples (X, y), where
            X is the video tensor and y is the corresponding label tensor.

    Each item from the dataloader will be saved as a .pt file in the specified
    directory, with filenames formatted as 'tensor_<index>.pt'.
    The directory will be created if it does not already exist.
    """
    for n, (X, y) in enumerate(dataloader):
        tensor_data = {
            'video': X,
            'label': y
        }
        os.makedirs(path, exist_ok=True)
        torch.save(tensor_data, os.path.join(path, f'tensor_{n}.pt'))

def pickle_data(X : torch.tensor, y : torch.tensor, n : int, path : str):
    """
    Saves the input data and labels as a PyTorch tensor file.

    Args:
        X (tensor): The input data to be saved (e.g., video frames, features).
        y (tensor): The labels corresponding to the input data.
        n (int): A numerical identifier that will be used to name the saved file.
        path (str): The directory path where the tensor file will be saved.

    Saves:
        A tensor file named 'tensor_{n}.pt' containing the input data (X) and labels (y) 
        at the specified path. The directory is created if it doesn't exist.

    Example:
        pickle_data(X_data, y_labels, 1, '/path/to/save')
    """
    tensor_data = {
        'video': X,
        'label': y
    }
    os.makedirs(path, exist_ok=True)
    torch.save(tensor_data, os.path.join(path, f'tensor_{n}.pt'))

def read_pickle_dict(path) -> dict:
    """
    Loads a PyTorch tensor file and returns the raw dict.

    Args:
        path (str): The file path to the tensor file to be loaded.

    Returns:
        tuple: A dictionary containing:
            - video (torch.tensor): The input data (e.g., video frames, features).
            - label (torch.tensor): The corresponding labels for the input data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.

    Example:
        tensor = read_pickle('/path/to/tensor_file.pt')
    """
    return torch.load(path)

def read_pickle(path : str):
    """
    Loads a PyTorch tensor file and returns the input data and labels.

    Args:
        path (str): The file path to the tensor file to be loaded.

    Returns:
        tuple: A tuple containing:
            - video (torch.tensor): The input data (e.g., video frames, features).
            - label (torch.tensor): The corresponding labels for the input data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        KeyError: If the tensor file does not contain the expected 'video' and 'label' keys.

    Example:
        X, y = read_pickle('/path/to/tensor_file.pt')
    """
    temp = torch.load(path)
    return temp['video'], temp['label']

def pickle_files_path_list(path) -> list:
    """
    Recursively collects all file paths in a specified directory.

    Args:
        path (str): The root directory to start searching for files.

    Returns:
        list: A list of file paths (str) for all files in the directory and its subdirectories.

    Example:
        file_paths = pickle_files_path_list('/path/to/directory')
    """
    path_list = []
    for top, dirs, files in os.walk(path):
        for nm in files:       
            path_list.append(os.path.join(top, nm))
    return path_list

def preload_tensors(path_list, device):
    """
    Loads tensors from a list of file paths and transfers them to the specified device.

    Args:
        path_list (list): A list of file paths to the tensor files to be loaded.
        device (torch.device): The device (CPU or CUDA) to which the tensors will be transferred.

    Returns:
        list: A list of tuples, where each tuple contains:
            - video (torch.tensor): The input data (e.g., video frames, features) loaded and transferred to the specified device.
            - label (torch.tensor): The corresponding labels, loaded and transferred to the specified device.

    Example:
        data = preload_tensors(['/path/to/file1.pt', '/path/to/file2.pt'], torch.device('cuda'))
    """
    data = []
    for path in path_list:
        tensor = read_pickle_dict(path)  # Load tensor dict from disk
        video, label = tensor['video'].to(device), tensor['label'].to(device) # Move the tensors to device
        data.append((video, label))
    return data