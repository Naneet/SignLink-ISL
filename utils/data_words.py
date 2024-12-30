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
