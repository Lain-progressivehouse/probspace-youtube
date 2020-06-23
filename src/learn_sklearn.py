import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

from src import data_frame, learn_lgb
from sklearn.linear_model import Ridge


def preprocess(train_x, test_x):
    cols = train_x.keys()
    sc = StandardScaler()
    sc.fit(train_x)
    return pd.DataFrame(sc.transform(train_x), columns=cols), pd.DataFrame(sc.transform(test_x), columns=cols)


def predict_cv(train_x, train_y, test_x, seed=22):
    preds = []
    preds_test = []
    va_indxes = []
    importance = pd.DataFrame()
    sss = KFold(n_splits=5, shuffle=True, random_state=seed)
    # sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # sss = GroupKFold(n_splits=10)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(sss.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model = Ridge(alpha=1.0)
        model.fit(tr_x, tr_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_indxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べなおす
    va_indxes = np.concatenate(va_indxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_indxes)
    pred_train = preds[order]
    # テストデータに対する予測値の平均をとる
    pred_test = np.mean(preds_test, axis=0)

    return pred_train, pred_test


def main(train_x=None, train_y=None, test_x=None, ids=None, seed=22):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    drop_null = set(test_x.keys()[test_x.isna().any()].to_list() + train_x.keys()[train_x.isna().any()].to_list())
    drop_list = ["publishedAt", "categoryId", "collection_date"] + list(drop_null)
    train_x = train_x.drop(drop_list, axis=1)
    test_x = test_x.drop(drop_list, axis=1)

    # train_x, test_x = preprocess(train_x, test_x)

    pred_train, pred_test = predict_cv(train_x, train_y, test_x, seed=seed)

    learn_lgb.output_metrics(train_y, pred_train)

    # sub = pd.DataFrame()
    # sub["id"] = ids
    # sub['y'] = np.expm1(pred_test)
    #
    # sub.to_csv('./data/output/ridge.csv', index=False)

    return pred_train, pred_test


def ensemble():
    train_x, train_y, test_x, ids = data_frame.main()

    preds_train = []
    preds_test = []
    for i in range(5):
        pred_train, pred_test = main(train_x, train_y, test_x, ids, i)
        preds_train.append(pred_train)
        preds_test.append(pred_test)

    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    learn_lgb.output_metrics(train_y, pred_train)
    learn_lgb.output_metrics(np.expm1(train_y), np.expm1(pred_train))

    sub = pd.DataFrame()
    sub["id"] = ids
    sub['y'] = np.expm1(pred_test)

    sub.to_csv('./data/output/ridge.csv', index=False)
