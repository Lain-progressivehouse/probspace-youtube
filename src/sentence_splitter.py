from transformers import BertJapaneseTokenizer
import pandas as pd
import numpy as np

tokenizer = BertJapaneseTokenizer.from_pretrained("./data/model/bert")


def splitter(sentence):
    return tokenizer.tokenize(sentence)
