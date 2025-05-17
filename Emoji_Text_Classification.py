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
        
        punctuation_marks = [".", ",", "'"]
        for punctuation_mark in punctuation_marks:
            sentence = sentence.replace(punctuation_mark, "")

        words = sentence.strip().split()

        sum_vectors = np.zeros((self.dimension, ))
        for word in words:
            sum_vectors += word_vectors[word]

        avg_vector = sum_vectors / len(words)
        return avg_vector

    def load_model(self):
        self.model = tf.keras.models.Sequential([
        # tf.keras.layers.Dropout(0.4, input_shape=(self.dimension,)),
        tf.keras.layers.Dense(5, input_shape=(self.dimension, ), activation="softmax")
        ])

        return self.model

    def train(self, X_train, Y_train, word_vectors):

        X_train_avg = []

        for x_train in X_train:
            X_train_avg.append(self.sentence_to_feature_vectors_avg(x_train, word_vectors))

        X_train_avg = np.array(X_train_avg)
        Y_train_one_hot = tf.keras.utils.to_categorical(Y_train, num_classes=5)

        self.model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )
        self.model.fit(X_train_avg, Y_train_one_hot, epochs=300)

    def test(self, X_test, Y_test, word_vectors):

        X_test_avg = []

        for x_test in X_test:
            X_test_avg.append(self.sentence_to_feature_vectors_avg(x_test, word_vectors))

        X_test_avg = np.array(X_test_avg)
        Y_test_one_hot = tf.keras.utils.to_categorical(Y_test, num_classes=5)

        scores = self.model.evaluate(X_test_avg, Y_test_one_hot)
        print(scores)

    def predict(self, sentence, word_vectors):
        sentence_avg = self.sentence_to_feature_vectors_avg(sentence, word_vectors)
        sentence_avg = np.array([sentence_avg])
        result = self.model.predict(sentence_avg)
        y_pred = np.argmax(result)
        emoji = self.label_to_emoji(y_pred)
        print(f"\nThe emoji related to sentence '{sentence}' is: {emoji}")

    def label_to_emoji(self, label):
        emojies = ["❤️", "🏀", "😀", "😔", "🍴"]
        return emojies[label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, help="Write a sentence.")
    parser.add_argument("--dimension", type=int, 
                        help="Enter a preferred dimension for feature vectors (50, 100, 200 or 300).")

    arg = parser.parse_args()

    obj = EmojiTextClassifier(arg)

    X_train, Y_train = obj.load_dataset("dataset/train.csv")
    X_test, Y_test  = obj.load_dataset("dataset/test.csv")

    pre_trained_vectors_path = f"glove.6B/glove.6B.{arg.dimension}d.txt"
    word_vectors = obj.load_feature_vectors(vectors_path=pre_trained_vectors_path)

    obj.load_model()
    obj.train(X_train, Y_train, word_vectors)
    obj.test(X_test, Y_test, word_vectors)

    total_time = 0
    input_sentence = arg.sentence
    
    start = time.time()
    obj.predict(input_sentence, word_vectors)
    inference_time = time.time() - start

    print("Inference Time:", inference_time)