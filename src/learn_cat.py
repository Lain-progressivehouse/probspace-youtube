import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
import catboost
from src import data_frame, feature_selection, pseudo, learn_lgb

categorical_features = ['categoryId', 'comments_disabled', 'ratings_disabled', "year", "hour"]
params = {
    "use_best_model": True,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.04,
    "iterations": 10000,
    "depth": 5,
    "random_seed": 191,
    "early_stopping_rounds": 50
}


def predict_cv(params, train_x, train_y, test_x, seed=22):
    preds = []
    preds_test = []
    va_indxes = []
    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for i, (tr_idx, va_idx) in enumerate(sss.split(train_x, train_y // 2)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model = catboost.CatBoostRegressor(**params)
        model.fit(tr_x, tr_y,
                  cat_features=categorical_features,
                  eval_set=(va_x, va_y),
                  verbose=100,
                  use_best_model=True,
                  plot=False)

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


def ensemble(train_x=None, train_y=None, test_x=None, ids=None):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    preds_train = []
    preds_test = []
    for i in range(5):
        params["random_seed"] = i
        params["depth"] = 5 + i
        pred_train, pred_test = predict_cv(params, train_x, train_y, test_x, seed=i + 100)
        preds_train.append(pred_train)
        preds_test.append(pred_test)
        learn_lgb.output_metrics(train_y, pred_train)

    pred_train = np.mean(preds_train, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    learn_lgb.output_metrics(train_y, pred_train)
    learn_lgb.output_metrics(np.expm1(train_y), np.expm1(pred_train))

    return pred_train, pred_test


def em(complement=True):
    train_x, train_y, test_x, ids = data_frame.main(complement=complement)
    embeded_lgb_feature = feature_selection.null_importance(train_x, train_y, test_x, ids, create=False)
    if "categoryId_TE_mean" not in embeded_lgb_feature:
        embeded_lgb_feature.append("categoryId_TE_mean")
    if "ratings_disabled" not in embeded_lgb_feature:
        embeded_lgb_feature.append("ratings_disabled")
    train_x = train_x[embeded_lgb_feature]
    test_x = test_x[embeded_lgb_feature]

    train_x, train_y, test_x = pseudo.get_pseudo_data_set(train_x, train_y, test_x, threshold=0.3)
    pred_train, pred_test = ensemble(train_x, train_y, test_x, ids)

    sub = pd.DataFrame()
    sub["id"] = ids
    sub['y'] = np.expm1(pred_test)

    sub.to_csv(f'./data/output/test_cat_complement_{complement}.csv', index=False)

    sub = pd.DataFrame()
    sub['y'] = np.expm1(pred_train)

    sub.to_csv(f'./data/output/train_cat_complement_{complement}.csv', index=False)
