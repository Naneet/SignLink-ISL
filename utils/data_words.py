import os
import numpy as np
def data_and_words(path):
    """
    Loads file paths and their corresponding labels from a dataset directory structure.

    Args:
        path (str): Path to the root directory of the dataset. 
                    Each subdirectory should represent a label and contain files belonging to that category.

    Returns:
        tuple: A tuple containing:
            - data (list): A list of tuples where each tuple consists of:
                - The file path (str) to a data sample (video).
                - The numeric label (int) corresponding to the category.
            - words_list (list): A list of category names (str) corresponding to their numeric labels.
            - words_dict (dict): A dictionary mapping category names (str) to their numeric labels (int).

    Notes:
        - The numeric labels are assigned in the order of the subdirectories as returned by `os.listdir`.
        - Ensure that the directory structure is consistent, with subdirectories representing words
          and containing relevant files.

    Example:
        If the directory structure is:
        root/
        ├── cat/
        │   ├── cat1.mp4
        │   ├── cat2.mp4
        ├── dog/
        │   ├── dog1.mp4

        Calling `data_and_words("root")` will return:
            - data: [("root/cat/cat1.mp4", 0), ("root/cat/cat2.mp4", 0), ("root/dog/dog1.mp4", 1)]
            - words_list: ["cat", "dog"]
            - words_dict: {"cat": 0, "dog": 1}
    """

    words = os.listdir(path)
    words_list = []
    data = []
    for word in words:
        words_list.append(word)
        word_videos_path = os.path.join(path, word)
        videos = os.listdir(word_videos_path)
        for video in videos:
            data.append((os.path.join(word_videos_path, video), len(words_list)-1))

    words_dict = dict({})
    for num, word in enumerate(words_list):
        words_dict[word] = num

    return data, words_list, words_dict  # data will return a list of tuples where data[0] is the path to the video and data[1] is the label
                                         # words_list can be used to convert index to word
                                         # words_dict can be used to convert word to index


def random_split(data_dict, word_to_idx, train_ratio=0.8, seed=42):
    """
    Splits a dataset into training and testing sets based on the specified ratio.

    Args:
        data_dict (dict): A dictionary where keys are labels (e.g., strings), and values are lists of paths 
                          (e.g., file paths or data samples corresponding to those labels).
        train_ratio (float, optional): The proportion of data to include in the training set. Defaults to 0.8.
        seed (int, optional): Random seed for reproducibility of the data split. Defaults to 42.

    Returns:
        tuple: Two lists, `train_data` and `test_data`.
            - train_data: A list of tuples, where each tuple contains a path and its corresponding label index, 
                          e.g., [(path1, label_index1), (path2, label_index2), ...].
            - test_data: A list of tuples, similar to `train_data`, but for the testing set.

    Notes:
        - Paths for each label are shuffled before splitting to ensure randomness.

    Example:
        data_dict = {
            "cat": ["cat1.jpg", "cat2.jpg", "cat3.jpg"],
            "dog": ["dog1.jpg", "dog2.jpg"]
        }
        word_to_idx = {"cat": 0, "dog": 1}
        
        train_data, test_data = random_split(data_dict, train_ratio=0.7)
        # train_data: [("cat1.jpg", 0), ("dog1.jpg", 1), ...]
        # test_data: [("cat3.jpg", 0), ("dog2.jpg", 1)]
    """
    np.random.seed(seed)
    train_data = []
    test_data = []

    for label, paths in data_dict.items():
        # Shuffle the paths
        paths = np.array(paths)
        np.random.shuffle(paths)

        # Split into training and testing
        split_idx = int(len(paths) * train_ratio)
        train_paths = paths[:split_idx].tolist()
        test_paths = paths[split_idx:].tolist()

        # Append the paths with their labels
        train_data.extend([(path, word_to_idx[label]) for path in train_paths])
        test_data.extend([(path, word_to_idx[label]) for path in test_paths])

    return train_data, test_data
