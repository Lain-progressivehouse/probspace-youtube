import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.model_selection import KFold

from src import data_frame

params = {'num_leaves': 86,
          'objective': 'regression',
          'max_depth': 7,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "feature_fraction": 0.2,
          "bagging_seed": 11,
          "metrics": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.1,
          # 'early_stopping_rounds': 200,
          # 'n_estimators': 50000,
          'random_state': 0,
          }


def main(train_x=None, train_y=None, test_x=None, ids=None, seed=22):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    lgbr = LGBMRegressor(n_estimators=500, learning_rate=0.1, num_leaves=86, subsample_freq=1,
                         subsample=0.9, feature_fraction=0.2, bagging_seed=11, metrics="rmse",
                         reg_alpha=0.1, reg_lambda=0.1, random_state=0)
    embeded_lgb_selector = SelectFromModel(lgbr, threshold='1.25*median')
    embeded_lgb_selector.fit(train_x, train_y)

    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = train_x.loc[:, embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_feature)), 'selected features')
    return embeded_lgb_feature


def select_k_best(train_x=None, train_y=None, test_x=None, ids=None, seed=22, k=300):
    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    selector = SelectKBest(score_func=f_regression, k=5)
    selector.fit(train_x, train_y)
    mask = selector.get_support()
    return selector.transform(train_x), selector.transform(test_x)


def get_feature_importances(train_x, train_y, shuffle=False, seed=22):
    # 必要ならば目的変数をシャッフル
    if shuffle:
        train_y = pd.Series(np.random.permutation(train_y))

    importance = pd.DataFrame()
    importance["feature"] = train_x.keys()
    importance["importance"] = 0

    sss = KFold(n_splits=3, shuffle=True, random_state=seed)
    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    dtrain = lgb.Dataset(train_x, train_y)
    # dtest = lgb.Dataset(va_x, va_y, reference=dtrain)
    model = lgb.train(params, dtrain, 400, verbose_eval=100)
    importance["importance"] += model.feature_importance()

    # 特徴量の重要度を含むデータフレームを作成
    return importance.sort_values("importance", ascending=False)


def display_distributions(actual_imp_df, null_imp_df, feature):
    # ある特徴量に対する重要度を取得
    actual_imp = actual_imp_df.query(f"feature == '{feature}'")["importance"].mean()
    null_imp = null_imp_df.query(f"feature == '{feature}'")["importance"]

    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    a = ax.hist(null_imp, label="Null importances")
    ax.vlines(x=actual_imp, ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
    ax.legend(loc="upper right")
    ax.set_title(f"Importance of {feature.upper()}", fontweight='bold')
    plt.xlabel(f"Null Importance Distribution for {feature.upper()}")
    plt.ylabel("Importance")
    plt.show()


def null_importance(train_x=None, train_y=None, test_x=None, ids=None, seed=22, create=False):
    # 閾値を設定
    THRESHOLD = 40

    if not create:
        print(f"Create {create}")
        actual_importance = pd.read_csv("./data/null_importance.csv")
        imp_features = []
        for feature, score in zip(actual_importance["feature"], actual_importance["score"]):
            if score >= THRESHOLD:
                imp_features.append(feature)
        print(str(len(imp_features)), 'selected features')
        return imp_features

    if train_x is None:
        train_x, train_y, test_x, ids = data_frame.main()

    # 実際の目的変数でモデルを学習し、特徴量の重要度を含むデータフレームを作成
    actual_importance = get_feature_importances(train_x, train_y, shuffle=False)

    # 目的変数をシャッフルした状態でモデルを学習し、特徴量の重要度を含むデータフレームを作成
    N_RUNS = 100
    null_importance = pd.DataFrame()
    for i in range(N_RUNS):
        imp_df = get_feature_importances(train_x, train_y, shuffle=True, seed=i)
        imp_df["run"] = i + 1
        null_importance = pd.concat([null_importance, imp_df])

    # 実データにおいて特徴量の重要度が高かった上位5位を表示
    # for feature in actual_importance["feature"][:5]:
    #     display_distributions(actual_importance, null_importance, feature)



    score_list = []

    # 閾値を超える特徴量を取得
    imp_features = []
    for feature in actual_importance["feature"]:
        actual_value = actual_importance.query(f"feature=='{feature}'")["importance"].values
        null_value = null_importance.query(f"feature=='{feature}'")["importance"].values
        percentage = (null_value < actual_value).sum() / null_value.size * 100
        score_list.append(percentage)
        if percentage >= THRESHOLD:
            imp_features.append(feature)

    actual_importance["score"] = score_list
    actual_importance.to_csv("./data/null_importance.csv", index=False)

    print(str(len(imp_features)), 'selected features')
    return imp_features
