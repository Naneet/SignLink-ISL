import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def show_sequence(sequence, NUM_FRAMES):
    """
    Displays a sequence of frames in a grid format.

    Args:
        sequence (torch.tensor): A tensor containing frames to be displayed. 
                                        Each frame should be a 3D tensor (C, H, W) representing a color image.
        NUM_FRAMES (int): The total number of frames the tensor contains.

    Returns:
        None: The function directly displays the frames using `matplotlib`.

    Example:
        show_sequence(frame_sequence, 16)
    """
    columns = 4
    rows = (NUM_FRAMES + 1) // (columns)
    fig = plt.figure(figsize=(32, (16 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        frames = sequence[j].permute(1,2,0).numpy()
        frames = frames/ frames.max()
        plt.imshow(frames)
    plt.show()