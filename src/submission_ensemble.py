import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def main():
    submission_list = [
        "test_lgb_CV6976.csv",
        "submission_weight001.csv",
        "test_lgb_CV6989.csv",
        "test_lgb_CV7042.csv"
    ]
    submission_df_list = [pd.read_csv(f"./data/output/{file}") for file in submission_list]

    sub = pd.DataFrame()
    sub["id"] = submission_df_list[0]["id"]
    sub["y"] = np.expm1(np.mean([df.y.apply(np.log1p) for df in submission_df_list], axis=0))
    sub.to_csv("./data/output/ensemble_3.csv", index=False)
