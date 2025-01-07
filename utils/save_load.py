import torch

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model and optimizer state to a specified file.

    Args:
        state (dict): A dictionary containing the model state, optimizer state, and any other relevant training information.
        filename (str, optional): The name of the file where the checkpoint will be saved. Default is 'checkpoint.pth'.

    Example:
        save_checkpoint({'epoch': 10, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model_checkpoint.pth')
    """
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth") -> dict:
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        filename (str, optional): The name of the file from which the checkpoint will be loaded. Default is 'checkpoint.pth'.

    Returns:
        dict: A dictionary containing the model state, optimizer state, and any other relevant training information.

    Example:
        checkpoint = load_checkpoint('model_checkpoint.pth')\\
        model.load_state_dict(checkpoint['model_state_dict'])\\
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    """
    checkpoint = torch.load(filename)
    return checkpoint

def load_model(model, checkpoint):
    """
    Loads the model state from a checkpoint and updates the model.

    Args:
        model (torch.nn.Module): The model to load the state into.
        checkpoint (dict): A dictionary containing the model state, typically loaded from a checkpoint file.

    Returns:
        torch.nn.Module: The model with the loaded state.

    Example:
        checkpoint = load_checkpoint('model_checkpoint.pth')
        model = load_model(model, checkpoint)
    """
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
