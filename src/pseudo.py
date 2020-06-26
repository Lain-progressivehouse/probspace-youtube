import lightgbm as lgb
from optuna.integration import lightgbm_tuner
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats

from src import data_frame, feature_selection, learn_lgb


def get_predict_test(params, train_x, train_y, test_x, validation):
    preds_test = []
    for i, (tr_idx, va_idx) in enumerate(validation):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        dtrain = lgb.Dataset(tr_x, tr_y)
        dtest = lgb.Dataset(va_x, va_y, reference=dtrain)
        model = lgb.train(params, dtrain, 2000, valid_sets=dtest, verbose_eval=100)

        pred_test = model.predict(test_x)
        preds_test.append(pred_test)

    pred_test_mean = np.mean(preds_test, axis=0)
    pred_test_std = np.std(preds_test, axis=0)
    pred_test_std = stats.mstats.rankdata(pred_test_std) / test_x.shape[0]

    return pred_test_mean, pred_test_std


def get_pseudo_data_set(train_x, train_y, test_x: pd.DataFrame, threshold=0.2):
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    pred_test_mean, pred_test_std = get_predict_test(
        learn_lgb.params, train_x, train_y, test_x, sss.split(train_x, train_y // 4))

    train_x = pd.concat([train_x, test_x[pred_test_std < threshold].copy()]).reset_index(drop=True)
    train_y = pd.concat([train_y, pd.Series(pred_test_mean[pred_test_std < threshold])]).reset_index(drop=True)
    return train_x, train_y, test_x
