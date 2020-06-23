import re
import unicodedata
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src import sentence_splitter, data_frame, learn_sklearn, learn_lgb


def rating_dataset():
    # target: ["likes", "dislikes"]
    all_df = pd.concat(
        [pd.read_csv("./data/input/train_data.csv").drop(["y"], axis=1),
         pd.read_csv("./data/input/test_data.csv")]
    ).reset_index(drop=True)
    train = all_df[~all_df["ratings_disabled"] & ~all_df["comments_disabled"]].reset_index(drop=True)
    test = all_df[all_df["ratings_disabled"] & ~all_df["comments_disabled"]].reset_index(drop=True)
    test = test.drop(["likes", "dislikes"], axis=1)
    train.likes = train.likes.apply(np.log1p)
    train.dislikes = train.dislikes.apply(np.log1p)
    train.comment_count = train.comment_count.apply(np.log1p)
    test.comment_count = test.comment_count.apply(np.log1p)

    train["publishedAt"] = pd.to_datetime(train.publishedAt).apply(lambda x: x.value)
    test["publishedAt"] = pd.to_datetime(test.publishedAt).apply(lambda x: x.value)

    train["title_len"] = train.title.apply(lambda x: len(str(x)))
    test["title_len"] = test.title.apply(lambda x: len(str(x)))
    train["channelTitle_len"] = train.channelTitle.apply(lambda x: len(str(x)))
    test["channelTitle_len"] = test.channelTitle.apply(lambda x: len(str(x)))
    train["description_len"] = train.description.apply(lambda x: len(str(x)))
    test["description_len"] = test.description.apply(lambda x: len(str(x)))
    train["tags_count"] = train.tags.apply(lambda x: str(x).count("|"))
    test["tags_count"] = test.tags.apply(lambda x: str(x).count("|"))

    # 日本語を含むかかどうかの判定
    train["title_ja_count"] = train.title.apply(data_frame.is_japanese)
    test["title_ja_count"] = test.title.apply(data_frame.is_japanese)
    train["channelTitle_ja_count"] = train.channelTitle.apply(data_frame.is_japanese)
    test["channelTitle_ja_count"] = test.channelTitle.apply(data_frame.is_japanese)
    train["description_ja_count"] = train.description.apply(data_frame.is_japanese)
    test["description_ja_count"] = test.description.apply(data_frame.is_japanese)

    # アルファベットのカウント
    train["title_en_count"] = train.title.apply(data_frame.count_alphabet)
    test["title_en_count"] = test.title.apply(data_frame.count_alphabet)
    train["channelTitle_en_count"] = train.channelTitle.apply(data_frame.count_alphabet)
    test["channelTitle_en_count"] = test.channelTitle.apply(data_frame.count_alphabet)
    train["description_en_count"] = train.description.apply(data_frame.count_alphabet)
    test["description_en_count"] = test.description.apply(data_frame.count_alphabet)

    # 数字のカウント
    train["description_num_count"] = train.description.apply(data_frame.count_number)
    test["description_num_count"] = test.description.apply(data_frame.count_number)

    # urlのカウント
    train["description_url_count"] = train.description.apply(lambda x: str(x).count("://"))
    test["description_url_count"] = test.description.apply(lambda x: str(x).count("://"))

    all_df: pd.DataFrame = pd.concat(
        [train.drop(["likes", "dislikes"], axis=1), test], ignore_index=True).reset_index(drop=True)

    category = ["channelId", "categoryId", "collection_date"]
    target_list = ["comment_count", "title_len", "channelTitle_len", "description_len", "tags_count",
                   "description_ja_count", "description_en_count", "title_ja_count", "title_en_count",
                   "publishedAt"]
    for col, target in product(category, target_list):
        print(col, target)
        data_frame.group(train, test, col, target, all_df)

    data_frame.TE(train, test, "mean", train.likes, ["categoryId", "collection_date"])
    data_frame.TE(train, test, "std", train.likes, ["categoryId", "collection_date"])
    data_frame.TE(train, test, "mean", train.dislikes, ["categoryId", "collection_date"])
    data_frame.TE(train, test, "std", train.dislikes, ["categoryId", "collection_date"])

    return train, test


def comment_dataset():
    # target: ["comment_dataset"]
    all_df = pd.concat(
        [pd.read_csv("./data/input/train_data.csv").drop(["y"], axis=1),
         pd.read_csv("./data/input/test_data.csv")]
    ).reset_index(drop=True)
    train = all_df[~all_df["ratings_disabled"] & ~all_df["comments_disabled"]].reset_index(drop=True)
    test = all_df[~all_df["ratings_disabled"] & all_df["comments_disabled"]].reset_index(drop=True)
    test = test.drop(["comment_count"], axis=1)
    train.likes = train.likes.apply(np.log1p)
    train.dislikes = train.dislikes.apply(np.log1p)
    test.likes = test.likes.apply(np.log1p)
    test.dislikes = test.dislikes.apply(np.log1p)
    train.comment_count = train.comment_count.apply(np.log1p)

    train["publishedAt"] = pd.to_datetime(train.publishedAt).apply(lambda x: x.value)
    test["publishedAt"] = pd.to_datetime(test.publishedAt).apply(lambda x: x.value)

    train["title_len"] = train.title.apply(lambda x: len(str(x)))
    test["title_len"] = test.title.apply(lambda x: len(str(x)))
    train["channelTitle_len"] = train.channelTitle.apply(lambda x: len(str(x)))
    test["channelTitle_len"] = test.channelTitle.apply(lambda x: len(str(x)))
    train["description_len"] = train.description.apply(lambda x: len(str(x)))
    test["description_len"] = test.description.apply(lambda x: len(str(x)))
    train["tags_count"] = train.tags.apply(lambda x: str(x).count("|"))
    test["tags_count"] = test.tags.apply(lambda x: str(x).count("|"))

    # 日本語を含むかかどうかの判定
    train["title_ja_count"] = train.title.apply(data_frame.is_japanese)
    test["title_ja_count"] = test.title.apply(data_frame.is_japanese)
    train["channelTitle_ja_count"] = train.channelTitle.apply(data_frame.is_japanese)
    test["channelTitle_ja_count"] = test.channelTitle.apply(data_frame.is_japanese)
    train["description_ja_count"] = train.description.apply(data_frame.is_japanese)
    test["description_ja_count"] = test.description.apply(data_frame.is_japanese)

    # アルファベットのカウント
    train["title_en_count"] = train.title.apply(data_frame.count_alphabet)
    test["title_en_count"] = test.title.apply(data_frame.count_alphabet)
    train["channelTitle_en_count"] = train.channelTitle.apply(data_frame.count_alphabet)
    test["channelTitle_en_count"] = test.channelTitle.apply(data_frame.count_alphabet)
    train["description_en_count"] = train.description.apply(data_frame.count_alphabet)
    test["description_en_count"] = test.description.apply(data_frame.count_alphabet)

    # 数字のカウント
    train["description_num_count"] = train.description.apply(data_frame.count_number)
    test["description_num_count"] = test.description.apply(data_frame.count_number)

    # urlのカウント
    train["description_url_count"] = train.description.apply(lambda x: str(x).count("://"))
    test["description_url_count"] = test.description.apply(lambda x: str(x).count("://"))

    all_df: pd.DataFrame = pd.concat(
        [train.drop(["likes", "dislikes"], axis=1), test], ignore_index=True).reset_index(drop=True)

    category = ["channelId", "categoryId", "collection_date"]
    target_list = ["likes", "dislikes", "title_len", "channelTitle_len", "description_len", "tags_count",
                   "description_ja_count", "description_en_count", "title_ja_count", "title_en_count",
                   "publishedAt"]
    for col, target in product(category, target_list):
        data_frame.group(train, test, col, target, all_df)

    data_frame.TE(train, test, "mean", train.comment_count, ["categoryId", "collection_date"])
    data_frame.TE(train, test, "std", train.comment_count, ["categoryId", "collection_date"])

    return train, test


def rating_main():
    train, test = rating_dataset()
    drop_list = ["id", "video_id", "title", "channelId", "channelTitle",
                 "tags", "thumbnail_link", "description"]
    ids = test.video_id
    train_y_likes = train["likes"]
    train_y_dislikes = train["dislikes"]
    train_x = train.drop(drop_list + ["likes", "dislikes"], axis=1)
    test_x = test.drop(drop_list, axis=1)

    ensemble(train_x, train_y_likes, test_x, ids, "likes")
    ensemble(train_x, train_y_dislikes, test_x, ids, "dislikes")

    # return train_x, train_y_likes, train_y_dislikes, test_x, ids


def comment_main():
    train, test = comment_dataset()
    drop_list = ["id", "video_id", "title", "channelId", "channelTitle",
                 "tags", "thumbnail_link", "description"]
    ids = test.video_id
    train_y = train["comment_count"]
    train_x = train.drop(drop_list + ["comment_count"], axis=1)
    test_x = test.drop(drop_list, axis=1)

    ensemble(train_x, train_y, test_x, ids, "comment")

    # return train_x, train_y_likes, train_y_dislikes, test_x, ids


def ensemble(train_x, train_y, test_x, ids, name):
    preds_train = []
    preds_test = []
    for i in range(5):
        pred_train, pred_test = learn_sklearn.main(train_x, train_y, test_x, ids, i)
        preds_train.append(pred_train)
        preds_test.append(pred_test)

    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    learn_lgb.output_metrics(train_y, pred_train)
    learn_lgb.output_metrics(np.expm1(train_y), np.expm1(pred_train))

    sub = pd.DataFrame()
    sub["video_id"] = ids
    sub['y'] = np.expm1(pred_test)

    sub.to_csv(f'./data/input/complement_{name}.csv', index=False)
