import tensorflow as tf
import numpy as np
import pandas as pd


class EmojiTextClassifier:
    def __init__(self, pre_trained_vectors_path="/glove.6B/glove.6B.50d.txt"):
        self.pre_trained_vectors_path = pre_trained_vectors_path

    def load_feature_vectors(self, pre_trained_vectors_path):
        self.feature_vectors = open(pre_trained_vectors_path, encoding="utf-8")
        self.word_vectors = {}
        for line in self.feature_vectors:
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:], dtype=np.float64)
            self.word_vectors[word] = vector

    def load_dataset(self, file_path):
        df = pd.read_csv(file_path)
        self.X = np.array(df["sentence"])
        self.Y = np.array(df["label"], dtype=int)
        return self.X, self.Y

    def sentence_to_feature_vectors_avg(self, sentence):
            try:
                sentence = sentence.lower()
                words = sentence.strip().split(" ")

                sum_vectors = np.zeros((50, ))
                for word in words:
                    sum_vectors += self.word_vectors[word]

                avg_vector = sum_vectors / len(words)
                return avg_vector
            except:
                print(sentence)
                return None

    def load_model(self):
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, input_shape=(50,), activation="softmax")
        ])
        self.model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )

    def train(self, dataset_train_path="/dataset/train.csv"):
        self.X_train, self.Y_train = self.load_dataset(dataset_train_path)

        X_train_avg = []

        for x_train in self.X_train:
            X_train_avg.append(self.sentence_to_feature_vectors_avg(x_train))

        X_train_avg = np.array(X_train_avg)
        self.Y_train_one_hot = tf.keras.utils.to_categorical(self.Y_train, num_classes=5)


    def test(self, dataset_test_path="/dataset/test.csv"):
        self.X_test, self.Y_test = self.load_dataset(dataset_test_path)
        
    def predict(self):
        ...