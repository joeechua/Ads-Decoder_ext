import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import torch
import os
import re


class SentimentPreProcessor:

    def __init__(self, root="../data/annotations", embed_model="glove-wiki-gigaword-300"):
        self.embed_model = embed_model
        path = "./word2vecmodels/" + self.embed_model + ".model"
        if os.path.exists(path):
            self.model = KeyedVectors.load(path)
        else:
            self.model = api.load(self.embed_model)
            self.model.save(path)

        self.root = root

        # Embed size specifies the embedding size and is also the hidden size
        # of the first hidden layer of memory cells.
        self.embed_size = int(self.embed_model.split("-")[-1])

        self.id_to_word = {}
        self.word_to_id = {}
        filename = os.path.join(self.root, "sentiments_list.txt")
        # fill up dictionary from sentiments_list.txt file
        f = open(filename, "r")
        lines = f.readlines()
        for line in lines:
            sentiment = re.search("""(?<=ABBREVIATION: ").+(?=")""", line)
            index = re.search("""\d+(?=.)""", line)
            start_s, end_s = sentiment.span()
            start_i, end_i = index.span()

            # index
            id = int(line[start_i:end_i])
            # sentiment
            word = line[start_s:end_s]

            # add to dictionary
            self.id_to_word[id] = word
            self.word_to_id[word] = id

    def transform(self, target_lst):
        # flatten list
        lst = [item for sublist in target_lst for item in sublist]

        # convert to int
        lst = [int(num) for num in lst]

        most_common_descriptor = self.id_to_word[max(lst, key=lst.count)]

        try:
            vec = self.model.get_vector(most_common_descriptor)
        except KeyError:
            vec = np.zeros([self.embed_size])

        return torch.from_numpy(vec)


if __name__ == "__main__":
    s = SentimentPreProcessor()
    vec = s.transform([['14'], ['18'], ['14']])
    print(vec)
    print(type(vec))
