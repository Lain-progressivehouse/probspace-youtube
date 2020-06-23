import lightgbm as lgb
from optuna.integration import lightgbm_tuner
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

from src import data_frame, feature_selection


def augment(df_x, df_y):
    df = pd.concat([df_x, pd.DataFrame(df_y, columns=["y"])], axis=1)
    copy_df = df[~df["ratings_disabled"]].copy()
    copy_df["ratings_disabled"] = True
    copy_df["likes"] = 0
    copy_df["dislikes"] = 0
    copy_df["eval_count"] = df.likes + df.dislikes
    copy_df["likes_ratio"] = df.likes / df.eval_count
    copy_df["likes_ratio"].fillna(-1)
    copy_df["dislikes_ratio"] = df.dislikes / df.eval_count
    copy_df["dislikes_ratio"].fillna(-1)
    copy_df["score"] = df["comment_count"] * df["eval_count"]
    copy_df["score_2"] = df["comment_count"] / df["eval_count"]
    copy_df["likes"] = -1
    copy_df["dislikes"] = -1
    augment_df = pd.concat([df, copy_df]).reset_index(drop=True)
    augment_df_y = augment_df["y"]
    augment_df_x = augment_df.drop(["y"], axis=1)
    return augment_df_x, augment_df_y


def predict_cv(params, train_x, train_y, test_x, seed=22):
    preds = []
    preds_test = []
    va_indxes = []
    importance = pd.DataFrame()
    importance["column"] = train_x.keys()
    importance["importance"] = 0
    # sss = KFold(n_splits=5, shuffle=True, random_state=seed)
    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # sss = GroupKFold(n_splits=10)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(sss.split(train_x, train_y // 10)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # data augment
        # augment_tr_x, augment_tr_y = augment(tr_x, tr_y)

        dtrain = lgb.Dataset(tr_x, tr_y)
        dtest = lgb.Dataset(va_x, va_y, reference=dtrain)
        model = lgb.train(params, dtrain, 2000, valid_sets=dtest, verbose_eval=100)

        # best_params, history = dict(), list()
        # model = lightgbm_tuner.train(params, dtrain, num_boost_round=2000, valid_sets=dtest, verbose_eval=False,
        #                              best_params=best_params,
        #                              tuning_history=history, early_stopping_rounds=100)
        # print(best_params)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_indxes.append(va_idx)
        importance["importance"] += model.feature_importance()

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べなおす
    va_indxes = np.concatenate(va_indxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_indxes)
    pred_train = preds[order]
    # テストデータに対する予測値の平均をとる
    pred_test = np.mean(preds_test, axis=0)

    # print(importance.sort_values("importance", ascending=False))
    importance.sort_values("importance", ascending=False).to_csv("./data/importance.csv")

    return pred_train, pred_test


def output_metrics(y, pred_y):
    print('RMSE score = \t {}'.format(np.sqrt(mean_squared_error(y, pred_y))))
    print('MAE score = \t {}'.format(mean_absolute_error(y, pred_y)))


# params = {
#     'num_leaves': 86,
#     'objective': 'mean_squared_error',
#     'max_depth': 7,
#     'learning_rate': 0.1,
#     "boosting_type": "gbdt",
#     "subsample_freq": 1,
#     "subsample": 0.9,
#     "feature_fraction": 0.2,
#     "bagging_seed": 11,
#     "metric": 'rmse',
#     "verbosity": -1,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.1,
#     'early_stopping_rounds': 200,
#     'n_estimators': 50000,
#     'random_state': 0,
# }

params = {
    'objective': 'mean_squared_error',
    # 'max_depth': 6,
    'learning_rate': 0.1,
    "boosting_type": "gbdt",
    "metric": 'rmse',
    'lambda_l1': 3.1601163039739164e-06,
    'lambda_l2': 0.00029839724492614994,
    'num_leaves': 26,
    'feature_fraction': 0.6,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'random_state': 0,
    'early_stopping_rounds': 200,
    'n_estimators': 50000,
}


def main(train_x=None, train_y=None, test_x=None, ids=None, seed=22):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    pred_train, pred_test = predict_cv(params, train_x, train_y, test_x, seed)

    # output_metrics(train_y, pred_train)
    # output_metrics(np.expm1(train_y), np.expm1(pred_train))

    # sub = pd.DataFrame()
    # sub["id"] = ids
    # sub['y'] = np.expm1(pred_test)
    #
    # sub.to_csv('./data/output/lgb.csv', index=False)

    return pred_train, pred_test


def ensemble(train_x=None, train_y=None, test_x=None, ids=None):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    preds_train = []
    preds_test = []
    for i in range(5):
        params["random_state"] = i
        params["num_leaves"] += 1
        pred_train, pred_test = main(train_x, train_y, test_x, ids, i)
        preds_train.append(pred_train)
        preds_test.append(pred_test)

    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    output_metrics(train_y, pred_train)
    output_metrics(np.expm1(train_y), np.expm1(pred_train))

    # sub = pd.DataFrame()
    # sub["id"] = ids
    # sub['y'] = np.expm1(pred_test)
    #
    # sub.to_csv('./data/output/test_lgb.csv', index=False)
    #
    # sub = pd.DataFrame()
    # sub['y'] = np.expm1(pred_train)
    #
    # sub.to_csv('./data/output/train_lgb.csv', index=False)

    return pred_train, pred_test


def ensemble_div_period(train_x=None, train_y=None, test_x=None, ids=None):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    train_y = train_y / train_x.period.apply(np.log1p)

    preds_train = []
    preds_test = []
    for i in range(5):
        params["random_state"] = i
        params["num_leaves"] -= 1
        pred_train, pred_test = main(train_x, train_y, test_x, ids, i)
        preds_train.append(pred_train)
        preds_test.append(pred_test)

    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    train_y = train_y * train_x.period.apply(np.log1p)
    pred_train = pred_train * train_x.period.apply(np.log1p)
    pred_test = pred_test * test_x.period.apply(np.log1p)

    output_metrics(train_y, pred_train)
    output_metrics(np.expm1(train_y), np.expm1(pred_train))

    sub = pd.DataFrame()
    sub["id"] = ids
    sub['y'] = np.expm1(pred_test)

    sub.to_csv('./data/output/lgb.csv', index=False)

    return pred_train, pred_test


def ensemble_diff_category_TE(train_x=None, train_y=None, test_x=None, ids=None):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    train_y = train_y - train_x.categoryId_TE_mean

    preds_train = []
    preds_test = []
    for i in range(5):
        params["random_state"] = i
        # params["num_leaves"] -= 1
        pred_train, pred_test = main(train_x, train_y, test_x, ids, i)
        preds_train.append(pred_train)
        preds_test.append(pred_test)

    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    train_y = train_y + train_x.categoryId_TE_mean
    pred_train = pred_train + train_x.categoryId_TE_mean
    pred_test = pred_test + test_x.categoryId_TE_mean

    output_metrics(train_y, pred_train)
    output_metrics(np.expm1(train_y), np.expm1(pred_train))

    return pred_train, pred_test


def em(complement=True):
    train_x, train_y, test_x, ids = data_frame.main(complement=complement)
    embeded_lgb_feature = feature_selection.null_importance(train_x, train_y, test_x, ids)
    # embeded_lgb_feature = feature_selection.main(train_x, train_y, test_x, ids)
    if "categoryId_TE_mean" not in embeded_lgb_feature:
        embeded_lgb_feature.append("categoryId_TE_mean")
    if "ratings_disabled" not in embeded_lgb_feature:
        embeded_lgb_feature.append("ratings_disabled")

    preds_train = []
    preds_test = []
    # categoryIdのTEの差分
    pred_train, pred_test = ensemble_diff_category_TE(
        train_x[embeded_lgb_feature], train_y, test_x[embeded_lgb_feature], ids)
    preds_train.append(pred_train)
    preds_test.append(pred_test)

    # 通常
    pred_train, pred_test = ensemble(
        train_x[embeded_lgb_feature], train_y, test_x[embeded_lgb_feature], ids)
    preds_train.append(pred_train)
    preds_test.append(pred_test)

    # 期間で除算
    pred_train, pred_test = ensemble_div_period(
        train_x[embeded_lgb_feature], train_y, test_x[embeded_lgb_feature], ids)
    preds_train.append(pred_train)
    preds_test.append(pred_test)

    # 平均をとる
    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    output_metrics(train_y, pred_train)
    output_metrics(np.expm1(train_y), np.expm1(pred_train))
    sub = pd.DataFrame()
    sub["id"] = ids
    sub['y'] = np.expm1(pred_test)

    sub.to_csv('./data/output/test_lgb.csv', index=False)

    sub = pd.DataFrame()
    sub['y'] = np.expm1(pred_train)

    sub.to_csv('./data/output/train_lgb.csv', index=False)

    return pred_train, pred_test


def em2():
    train_x, train_y, test_x, ids = data_frame.main(complement=True)
    pred_train_a, pred_test_a = em(complement=True)
    pred_train_b, pred_test_b = em(complement=False)
    # 平均をとる
    pred_train = np.mean([pred_train_a, pred_train_b], axis=0)
    pred_test = np.mean([pred_test_a, pred_test_b], axis=0)

    # pred_train = []
    # for comments_disabled, ratings_disabled, pred_a, pred_b in zip(train_x["comments_disabled"],
    #                                                                train_x["ratings_disabled"], pred_train_a,
    #                                                                pred_train_b):
    #     if comments_disabled ^ ratings_disabled:
    #         pred_train.append(pred_a)
    #     else:
    #         pred_train.append(pred_b)
    #
    # pred_test = []
    # for comments_disabled, ratings_disabled, pred_a, pred_b in zip(test_x["comments_disabled"],
    #                                                                test_x["ratings_disabled"], pred_test_a,
    #                                                                pred_test_b):
    #     if comments_disabled ^ ratings_disabled:
    #         pred_test.append(pred_a)
    #     else:
    #         pred_test.append(pred_b)

    output_metrics(train_y, pred_train)
    output_metrics(np.expm1(train_y), np.expm1(pred_train))

    sub = pd.DataFrame()
    sub["id"] = ids
    sub['y'] = np.expm1(pred_test)

    sub.to_csv('./data/output/test_lgb.csv', index=False)

    sub = pd.DataFrame()
    sub['y'] = np.expm1(pred_train)

    sub.to_csv('./data/output/train_lgb.csv', index=False)
