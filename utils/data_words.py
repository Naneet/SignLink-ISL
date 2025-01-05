import os
def data_and_words(path):
    """Use this function to load the data path and label for the dataset"""
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


def random_split(data_dict, train_ratio=0.8, seed=42):
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
