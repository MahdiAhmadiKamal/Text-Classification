import time
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd


class EmojiTextClassifier:
    def __init__(self, arg):
        self.dimension = arg.dimension

    def load_dataset(self, file_path):
        df = pd.read_csv(file_path)
        X = np.array(df["sentence"])
        Y = np.array(df["label"], dtype=int)
        return X, Y

    def load_feature_vectors(self, vectors_path):
        feature_vectors = open(vectors_path, encoding="utf-8")
        word_vectors = {}
        for line in feature_vectors:
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:], dtype=np.float64)
            word_vectors[word] = vector
        return word_vectors

    def sentence_to_feature_vectors_avg(self, sentence, word_vectors):
        
        sentence = sentence.lower()
        sentence = sentence.replace(".", "")
        sentence = sentence.replace("'", "")
        words = sentence.strip().split()

        sum_vectors = np.zeros((self.dimension, ))
        for word in words:
            sum_vectors += word_vectors[word]

        avg_vector = sum_vectors / len(words)
        return avg_vector

   