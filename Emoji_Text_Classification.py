import time
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd


class EmojiTextClassifier:
    def __init__(self, arg):
        self.dimension = arg.dimension
