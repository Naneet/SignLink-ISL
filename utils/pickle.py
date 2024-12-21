import torch

def pickle_data(path, dataloader):
    for n, (X, y) in enumerate(dataloader):
        tensor_data = {
            'video': X,
            'label': y
        }
        
        torch.save(tensor_data, f'{path}\\tensor_{n}.pt')


def read_pickle(path) -> torch.tensor:
    return torch.load(path)

def pickle_files_path_list(path) -> list:
    path_list = []
    for top, dirs, files in os.walk(path):
        for nm in files:       
            path_list.append(os.path.join(top, nm))
    return path_list
