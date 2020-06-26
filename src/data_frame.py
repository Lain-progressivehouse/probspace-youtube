import re
import unicodedata
from collections import Counter
from itertools import product
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import umap
import pickle

from src import sentence_splitter


def get_umap(train, test, size=2):
    um = umap.UMAP(transform_seed=1, random_state=1, n_neighbors=size)
    um.fit(train.values)
    tr_em = um.transform(train.values)
    te_em = um.transform(test.values)
    return tr_em, te_em


def LE(train, test):
    for col in train.columns:
        if train[col].dtypes == object:
            train[col].fillna("null")
            test[col].fillna("null")
            lbl = LabelEncoder()
            lbl.fit(list(train[col].values) + list(test[col].values))
            train[col] = lbl.transform(list(train[col].values))
            test[col] = lbl.transform(list(test[col].values))


# カウントエンコーディング
def CE(train, test, cols, all_df):
    for col in cols:
        # all_df = pd.concat([train.drop(["y"], axis=1), test], ignore_index=True).reset_index()
        train[col + "_count"] = train[col].map(all_df[col].value_counts())
        test[col + "_count"] = test[col].map(all_df[col].value_counts())


# ターゲットエンコーディング
def TE(train, test, func, target, cols):
    funcs = ["max", "min", "mean", "std"]
    for col in cols:
        data_tmp = pd.DataFrame({col: train[col], "target": target})
        target_dic = data_tmp.groupby(col)["target"].aggregate(func)
        test[col + "_TE_" + func] = test[col].map(target_dic)

        tmp = np.repeat(np.nan, train.shape[0])

        # 学習データを分割
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
        for idx_1, idx_2 in kf.split(train, train[col]):
            target_dic = data_tmp.iloc[idx_1].groupby(col)["target"].aggregate(func)
            tmp[idx_2] = train[col].iloc[idx_2].map(target_dic)

        train[col + "_TE_" + func] = tmp


def group(train, test, col, target, all_df):
    mean_map = all_df.groupby(col)[target].mean()
    train["group_" + col + "_mean_" + target] = train[col].map(mean_map)
    test["group_" + col + "_mean_" + target] = test[col].map(mean_map)
    std_map = all_df.groupby(col)[target].std()
    train["group_" + col + "_std_" + target] = train[col].map(std_map)
    test["group_" + col + "_std_" + target] = test[col].map(std_map)
    sum_map = all_df.groupby(col)[target].sum()
    train["group_" + col + "_sum_" + target] = train[col].map(sum_map)
    test["group_" + col + "_sum_" + target] = test[col].map(sum_map)
    min_map = all_df.groupby(col)[target].min()
    train["group_" + col + "_min_" + target] = train[col].map(min_map)
    test["group_" + col + "_min_" + target] = test[col].map(min_map)
    max_map = all_df.groupby(col)[target].max()
    train["group_" + col + "_max_" + target] = train[col].map(max_map)
    test["group_" + col + "_max_" + target] = test[col].map(max_map)


def calculate(df: pd.DataFrame):
    df["eval_count"] = df.likes + df.dislikes
    df["likes_ratio"] = df.likes / df.eval_count
    df["likes_ratio"].fillna(-1)
    df["dislikes_ratio"] = df.dislikes / df.eval_count
    df["dislikes_ratio"].fillna(-1)
    df["score"] = df["comment_count"] * df["eval_count"]
    df["score_2"] = df["comment_count"] / df["eval_count"]
    df["title_div_description"] = df["title_len"] / df["description_len"]
    df["title_mul_description"] = df["title_len"] * df["description_len"]


def is_japanese(string):
    count = 0
    for ch in str(string):
        try:
            name = unicodedata.name(ch)
        except:
            continue
        if "CJK UNIFIED" in name \
                or "HIRAGANA" in name \
                or "KATAKANA" in name:
            count += 1
    return count


def count_alphabet(string):
    r = re.compile(r"[a-z|A-Z]+")
    return len("".join(r.findall(str(string))))


def count_number(string):
    r = re.compile(r"[0-9]+")
    return len("".join(r.findall(str(string))))


def change_to_Date(train, test, input_column_name, output_column_name):
    train[output_column_name] = train[input_column_name].map(lambda x: x.split('.'))
    test[output_column_name] = test[input_column_name].map(lambda x: x.split('.'))
    train[output_column_name] = train[output_column_name].map(
        lambda x: '20' + x[0] + '-' + x[2] + '-' + x[1] + 'T00:00:00.000Z')
    test[output_column_name] = test[output_column_name].map(
        lambda x: '20' + x[0] + '-' + x[2] + '-' + x[1] + 'T00:00:00.000Z')


def tag_counter(train, test, n=500, pca_size=None, drop=False, create=True):
    cols = [f"tags_{i}" for i in range(n)]
    if create:
        # tagのカウント
        tags = []
        for tag in train["tags"]:
            tags.extend(str(tag).split("|"))
        tmp = Counter(tags)
        tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:n]
        for i, item in enumerate(tmp):
            train[f"tags_{i}"] = train["tags"].apply(lambda x: 1 if item[0] in str(x).split("|") else 0)
            test[f"tags_{i}"] = test["tags"].apply(lambda x: 1 if item[0] in str(x).split("|") else 0)

        train[cols].to_csv("./data/input/train_tags.csv", index=False)
        test[cols].to_csv("./data/input/test_tags.csv", index=False)
    else:
        train_tags = pd.read_csv("./data/input/train_tags.csv")
        test_tags = pd.read_csv("./data/input/test_tags.csv")
        train = pd.concat([train, train_tags[cols]], axis=1)
        test = pd.concat([test, test_tags[cols]], axis=1)

    if pca_size:
        # pca = TruncatedSVD(n_components=pca_size, random_state=2)
        # pca.fit(train[cols])
        # train_pca = pca.transform(train[cols])
        # test_pca = pca.transform(test[cols])
        train_pca, test_pca = get_umap(train[cols], test[cols], size=pca_size)
        pca_cols = [f"tangs_pca_{i}" for i in range(pca_size)]
        train = pd.concat([train, pd.DataFrame(train_pca, columns=pca_cols)], axis=1)
        test = pd.concat([test, pd.DataFrame(test_pca, columns=pca_cols)], axis=1)
        if drop:
            train = train.drop(cols, axis=1)
            test = test.drop(cols, axis=1)

    return train, test


def title_counter(train, test, n=100, pca_size=None, drop=False, create=True):
    train["title_words"] = train.title.apply(lambda x: sentence_splitter.splitter(str(x)))
    test["title_words"] = test.title.apply(lambda x: sentence_splitter.splitter(str(x)))
    cols = [f"title_word_{i}" for i in range(n)]
    if create:
        # titleの単語のカウント
        word_list = []
        for words in train["title_words"]:
            word_list.extend(words)
        tmp = Counter(word_list)
        tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:n]
        for i, item in enumerate(tmp):
            train[f"title_word_{i}"] = train["title_words"].apply(lambda x: x.count(item[0]))
            test[f"title_word_{i}"] = test["title_words"].apply(lambda x: x.count(item[0]))

        train[cols].to_csv("./data/input/train_title_words.csv", index=False)
        test[cols].to_csv("./data/input/test_title_words.csv", index=False)
    else:
        train_tags = pd.read_csv("./data/input/train_title_words.csv")
        test_tags = pd.read_csv("./data/input/test_title_words.csv")
        train = pd.concat([train, train_tags[cols]], axis=1)
        test = pd.concat([test, test_tags[cols]], axis=1)

    if pca_size:
        # pca = TruncatedSVD(n_components=pca_size, random_state=2)
        # pca.fit(train[cols])
        # train_pca = pca.transform(train[cols])
        # test_pca = pca.transform(test[cols])
        train_pca, test_pca = get_umap(train[cols], test[cols], size=pca_size)
        pca_cols = [f"title_pca_{i}" for i in range(pca_size)]
        train = pd.concat([train, pd.DataFrame(train_pca, columns=pca_cols)], axis=1)
        test = pd.concat([test, pd.DataFrame(test_pca, columns=pca_cols)], axis=1)
        if drop:
            train = train.drop(cols, axis=1)
            test = test.drop(cols, axis=1)

    train = train.drop(["title_words"], axis=1)
    test = test.drop(["title_words"], axis=1)
    return train, test


def category_unstack(train, test, all_df, group, category, normalize=True, pca_size=2):
    use_columns = set(train[category].unique()) & set(test[category].unique())
    unstack_df = all_df.groupby(group)[category].value_counts(normalize=normalize).unstack().fillna(0)
    for col in use_columns:
        train[f"{category}_{col}_ratio_in_{group}_group"] = train[group].map(unstack_df[col])
        test[f"{category}_{col}_ratio_in_{group}_group"] = test[group].map(unstack_df[col])

    cols = [f"{category}_{col}_ratio_in_{group}_group" for col in use_columns]
    pca_cols = [f"{category}_pca_{i}_in_{group}_group" for i in range(pca_size)]
    pca = TruncatedSVD(n_components=pca_size, random_state=2)
    pca.fit(train[cols])
    train_pca = pca.transform(train[cols])
    test_pca = pca.transform(test[cols])
    train = pd.concat([train, pd.DataFrame(train_pca, columns=pca_cols)], axis=1)
    test = pd.concat([test, pd.DataFrame(test_pca, columns=pca_cols)], axis=1)
    return train, test


def make_dataset(complement=True):
    train = pd.read_csv("./data/input/train_data.csv")
    test = pd.read_csv("./data/input/test_data.csv")

    if complement:
        complement_likes = pd.read_csv("./data/input/complement_likes.csv")
        complement_dislikes = pd.read_csv("./data/input/complement_dislikes.csv")
        complement_comment = pd.read_csv("./data/input/complement_comment.csv")
        likes_dict = dict(zip(complement_likes.video_id, complement_likes.y))
        dislikes_dict = dict(zip(complement_dislikes.video_id, complement_dislikes.y))
        comment_dict = dict(zip(complement_comment.video_id, complement_comment.y))
        train["likes"] = train.apply(
            lambda x: likes_dict[x["video_id"]] if x["video_id"] in likes_dict.keys() else x["likes"], axis=1)
        train["dislikes"] = train.apply(
            lambda x: dislikes_dict[x["video_id"]] if x["video_id"] in dislikes_dict.keys() else x["dislikes"], axis=1)
        train["comment_count"] = train.apply(
            lambda x: comment_dict[x["video_id"]] if x["video_id"] in comment_dict.keys() else x["comment_count"],
            axis=1)
        test["likes"] = test.apply(
            lambda x: likes_dict[x["video_id"]] if x["video_id"] in likes_dict.keys() else x["likes"], axis=1)
        test["dislikes"] = test.apply(
            lambda x: dislikes_dict[x["video_id"]] if x["video_id"] in dislikes_dict.keys() else x["dislikes"], axis=1)
        test["comment_count"] = test.apply(
            lambda x: comment_dict[x["video_id"]] if x["video_id"] in comment_dict.keys() else x["comment_count"],
            axis=1)

    # サムネイルの色の平均
    # train_thumbnail = pd.read_csv("./data/input/train_thumbnail.csv")
    # test_thumbnail = pd.read_csv("./data/input/test_thumbnail.csv")
    # train = train.merge(train_thumbnail, on="video_id")
    # test = test.merge(test_thumbnail, on="video_id")

    # サムネイル特徴量
    # train_image_features = pd.read_csv("./data/input/train_image_features.csv")
    # test_image_features = pd.read_csv("./data/input/test_image_features.csv")
    # train_umap, test_umap = get_umap(train_image_features, test_image_features, size=2)
    # pca_cols = [f"image_features_umap_{i}" for i in range(2)]
    # train = pd.concat([train, pd.DataFrame(train_umap, columns=pca_cols)], axis=1)
    # test = pd.concat([test, pd.DataFrame(test_umap, columns=pca_cols)], axis=1)

    train.likes = train.likes.apply(np.log1p)
    test.likes = test.likes.apply(np.log1p)
    train.dislikes = train.dislikes.apply(np.log1p)
    test.dislikes = test.dislikes.apply(np.log1p)
    train.comment_count = train.comment_count.apply(np.log1p)
    test.comment_count = test.comment_count.apply(np.log1p)

    train["title_len"] = train.title.apply(lambda x: len(str(x)))
    test["title_len"] = test.title.apply(lambda x: len(str(x)))
    train["channelTitle_len"] = train.channelTitle.apply(lambda x: len(str(x)))
    test["channelTitle_len"] = test.channelTitle.apply(lambda x: len(str(x)))
    train["description_len"] = train.description.apply(lambda x: len(str(x)))
    test["description_len"] = test.description.apply(lambda x: len(str(x)))
    train["tags_count"] = train.tags.apply(lambda x: str(x).count("|"))
    test["tags_count"] = test.tags.apply(lambda x: str(x).count("|"))

    # 時間系
    train["year"] = pd.to_datetime(train.publishedAt).apply(lambda x: x.year)
    test["year"] = pd.to_datetime(test.publishedAt).apply(lambda x: x.year)
    train["month"] = pd.to_datetime(train.publishedAt).apply(lambda x: x.month)
    test["month"] = pd.to_datetime(test.publishedAt).apply(lambda x: x.month)
    train["hour"] = pd.to_datetime(train.publishedAt).apply(lambda x: x.hour)
    test["hour"] = pd.to_datetime(test.publishedAt).apply(lambda x: x.hour)
    change_to_Date(train, test, "collection_date", "collectionAt")
    train["period"] = (pd.to_datetime(train.collectionAt) - pd.to_datetime(train.publishedAt)).apply(lambda x: x.days)
    test["period"] = (pd.to_datetime(test.collectionAt) - pd.to_datetime(test.publishedAt)).apply(lambda x: x.days)
    train["publishedAt"] = pd.to_datetime(train.publishedAt).apply(lambda x: x.value)
    test["publishedAt"] = pd.to_datetime(test.publishedAt).apply(lambda x: x.value)
    train["collectionAt"] = pd.to_datetime(train.collectionAt).apply(lambda x: x.value)
    test["collectionAt"] = pd.to_datetime(test.collectionAt).apply(lambda x: x.value)

    # 1日ごとの集計
    train["likes_per_day"] = train.likes.apply(np.expm1) / train.period
    test["likes_per_day"] = test.likes.apply(np.expm1) / test.period
    train["dislikes_per_day"] = train.dislikes.apply(np.expm1) / train.period
    test["dislikes_per_day"] = test.dislikes.apply(np.expm1) / test.period
    train["comment_per_day"] = train.comment_count.apply(np.expm1) / train.period
    test["comment_per_day"] = test.comment_count.apply(np.expm1) / test.period

    calculate(train)
    calculate(test)

    # 日本語を含むかかどうかの判定
    train["title_ja_count"] = train.title.apply(is_japanese)
    test["title_ja_count"] = test.title.apply(is_japanese)
    train["channelTitle_ja_count"] = train.channelTitle.apply(is_japanese)
    test["channelTitle_ja_count"] = test.channelTitle.apply(is_japanese)
    train["description_ja_count"] = train.description.apply(is_japanese)
    test["description_ja_count"] = test.description.apply(is_japanese)

    train["title_ja_ratio"] = train.title_ja_count / train.title_len
    test["title_ja_ratio"] = test.title_ja_count / test.title_len
    train["channelTitle_ja_ratio"] = train.channelTitle_ja_count / train.channelTitle_len
    test["channelTitle_ja_ratio"] = test.channelTitle_ja_count / test.channelTitle_len
    train["description_ja_ratio"] = train.title_ja_count / train.description_len
    test["description_ja_ratio"] = test.title_ja_count / test.description_len

    # アルファベットのカウント
    train["title_en_count"] = train.title.apply(count_alphabet)
    test["title_en_count"] = test.title.apply(count_alphabet)
    train["channelTitle_en_count"] = train.channelTitle.apply(count_alphabet)
    test["channelTitle_en_count"] = test.channelTitle.apply(count_alphabet)
    train["description_en_count"] = train.description.apply(count_alphabet)
    test["description_en_count"] = test.description.apply(count_alphabet)

    train["title_en_ratio"] = train.title_en_count / train.title_len
    test["title_en_ratio"] = test.title_en_count / test.title_len
    train["channelTitle_en_ratio"] = train.channelTitle_en_count / train.channelTitle_len
    test["channelTitle_en_ratio"] = test.channelTitle_en_count / test.channelTitle_len
    train["description_en_ratio"] = train.title_en_count / train.description_len
    test["description_en_ratio"] = test.title_en_count / test.description_len

    # 数字のカウント
    train["description_num_count"] = train.description.apply(count_number)
    test["description_num_count"] = test.description.apply(count_number)
    train["description_num_ratio"] = train.description_num_count / train.description_len
    test["description_num_ratio"] = test.description_num_count / test.description_len

    # urlのカウント
    train["description_url_count"] = train.description.apply(lambda x: str(x).count("://"))
    test["description_url_count"] = test.description.apply(lambda x: str(x).count("://"))

    if not complement:
        train.loc[train.ratings_disabled, "likes"] = -1
        train.loc[train.ratings_disabled, "dislikes"] = -1
        train.loc[train.comments_disabled, "comment_count"] = -1
        test.loc[test.ratings_disabled, "likes"] = -1
        test.loc[test.ratings_disabled, "dislikes"] = -1
        test.loc[test.comments_disabled, "comment_count"] = -1

    all_df: pd.DataFrame = pd.concat([train.drop(["y"], axis=1), test], ignore_index=True).reset_index(drop=True)
    train, test = category_unstack(train, test, all_df, "channelId", "categoryId")

    category = ["channelId", "categoryId", "collection_date"]
    CE(train, test, category, all_df)

    target_list = ["likes", "dislikes", "comment_count", "title_len", "channelTitle_len", "description_len",
                   "tags_count", "description_ja_count", "description_en_count", "title_ja_count", "title_en_count",
                   "publishedAt",
                   "likes_ratio", "period", "likes_ratio", "title_en_ratio", "description_ja_ratio",
                   "likes_per_day", "dislikes_per_day", "comment_per_day"]

    for col, target in product(category, target_list):
        group(train, test, col, target, all_df)

    # train["group_categoryId_mean_likes_ratio"] = train["likes"] / train["group_categoryId_mean_likes"]
    # train["group_categoryId_mean_likes_diff"] = train["likes"] - train["group_categoryId_mean_likes"]
    # train["group_categoryId_mean_dislikes_ratio"] = train["dislikes"] / train["group_categoryId_mean_dislikes"]
    # train["group_categoryId_mean_dislikes_diff"] = train["dislikes"] - train["group_categoryId_mean_dislikes"]
    #
    # train["group_channelId_mean_likes_ratio"] = train["likes"] / train["group_channelId_mean_likes"]
    # train["group_channelId_mean_likes_diff"] = train["likes"] - train["group_channelId_mean_likes"]
    # train["group_channelId_mean_dislikes_ratio"] = train["dislikes"] / train["group_channelId_mean_dislikes"]
    # train["group_channelId_mean_dislikes_diff"] = train["dislikes"] - train["group_channelId_mean_dislikes"]

    # BERT特徴量
    # is_pca = True
    # train_bert = pd.read_csv("./data/input/train_bert.csv")
    # test_bert = pd.read_csv("./data/input/test_bert.csv")
    # pca = PCA(n_components=10, random_state=1)
    # pca.fit(train_bert.values)
    # if is_pca:
    #     cols = [f"bert_pca_{i}" for i in range(10)]
    #     train_bert = pd.DataFrame(pca.transform(train_bert.values), columns=cols)
    #     test_bert = pd.DataFrame(pca.transform(test_bert.values), columns=cols)
    # train = pd.concat([train, train_bert], axis=1)
    # test = pd.concat([test, test_bert], axis=1)

    train, test = tag_counter(train, test, n=1000, pca_size=2, drop=False, create=False)

    train, test = title_counter(train, test, n=50, pca_size=2, drop=False, create=False)

    TE(train, test, "mean", train.y.apply(np.log1p), ["categoryId"])
    test["categoryId_TE_mean"].fillna(train.y.apply(np.log1p).mean(), inplace=True)

    train["channel_period"] = train.group_channelId_max_publishedAt - train.group_channelId_min_publishedAt
    test["channel_period"] = test.group_channelId_max_publishedAt - test.group_channelId_min_publishedAt

    # copy_train = train[~train["ratings_disabled"]].copy()
    # copy_train["ratings_disabled"] = True
    # copy_train["likes"] = -1
    # copy_train["dislikes"] = -1
    # train = pd.concat([train, copy_train]).reset_index(drop=True)

    # importance高いもの同士で演算
    calc_list = ["likes_ratio", "period", "likes_ratio", "title_en_ratio", "description_ja_ratio",
                 "likes_per_day", "dislikes_per_day", "comment_per_day"]
    for col_1, col_2 in product(calc_list, calc_list):
        if col_1 == col_2:
            continue
        if f"{col_2}_mul_{col_1}" not in train.keys():
            train[f"{col_1}_mul_{col_2}"] = train[col_1] * train[col_2]
            test[f"{col_1}_mul_{col_2}"] = test[col_1] * test[col_2]
        train[f"{col_1}_div_{col_2}"] = train[col_1] / train[col_2]
        test[f"{col_1}_div_{col_2}"] = test[col_1] / test[col_2]

    with open(f"./data/input/pkl/train_complement_{complement}.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(f"./data/input/pkl/test_complement_{complement}.pkl", "wb") as f:
        pickle.dump(test, f)

    return train, test


def main(complement=True):
    # train, test = make_dataset(complement=complement)
    with open(f"./data/input/pkl/train_complement_{complement}.pkl", "rb") as f:
        train = pickle.load(f)
    with open(f"./data/input/pkl/test_complement_{complement}.pkl", "rb") as f:
        test = pickle.load(f)

    drop_list = ["id", "video_id", "title", "channelId", "channelTitle",
                 "tags", "thumbnail_link", "description"]
    ids = test.id
    train_y = train.y.apply(np.log1p)
    train_x = train.drop(drop_list + ["y"], axis=1)
    test_x = test.drop(drop_list, axis=1)

    LE(train_x, test_x)

    return train_x, train_y, test_x, ids
