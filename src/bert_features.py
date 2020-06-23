from transformers import BertConfig, BertJapaneseTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

MODEL_NAME = "bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
config = BertConfig.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')
print(f"使用デバイス: {device}")
bert.to(device)

def tokenize(title, description, max_length=256):
    id_dict = tokenizer.encode_plus(str(title), str(description),
                                    max_length=max_length,
                                    pad_to_max_length=True)
    return id_dict["input_ids"], id_dict["attention_mask"]


def main():
    train = pd.read_csv("./data/input/train_data.csv")
    test = pd.read_csv("./data/input/test_data.csv")
    train[["TEXT", "MASK"]] = train.apply(lambda x: tokenize(x["title"], x["description"]), axis=1,
                                          result_type="expand")
    test[["TEXT", "MASK"]] = test.apply(lambda x: tokenize(x["title"], x["description"]), axis=1,
                                        result_type="expand")

    train_bert = get_features(train)
    test_bert = get_features(test)
    num = train_bert.shape[1]
    cols = [f"bert_{i}" for i in range(num)]
    train_bert_df = pd.DataFrame(train_bert, columns=cols)
    test_bert_df = pd.DataFrame(test_bert, columns=cols)
    train_bert_df.to_csv("./data/input/train_bert.csv", index=False)
    test_bert_df.to_csv("./data/input/test_bert.csv", index=False)


def get_features(df):
    ds = TensorDataset(torch.tensor(df["TEXT"], dtype=torch.int64),
                       torch.tensor(df["MASK"], dtype=torch.int64))
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    for input_ids, attention_mask in tqdm(dl):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = bert(input_ids=input_ids, attention_mask=attention_mask)
        output = output[0]
        output = output.to(cpu)
        preds.append(output.detach().clone().numpy())
    return np.concatenate(preds, axis=0)
