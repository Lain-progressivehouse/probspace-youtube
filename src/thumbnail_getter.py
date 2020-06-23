import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
import cv2


def save_thumbnail(video_id, thumbnail_link):
    sleep(0.1)
    response = requests.get(thumbnail_link)
    image = response.content
    with open(f"./data/thumbnail/{video_id}.jpg", "wb") as aaa:
        aaa.write(image)


def main():
    train = pd.read_csv("./data/input/train_data.csv")
    test = pd.read_csv("./data/input/test_data.csv")
    all_df: pd.DataFrame = pd.concat([train.drop(["y"], axis=1), test], ignore_index=True).reset_index(drop=True)
    for video_id, thumbnail_link in tqdm(zip(all_df["video_id"], all_df["thumbnail_link"]), total=len(all_df)):
        save_thumbnail(video_id, thumbnail_link)


def get_grb_hsv(video_id):
    img = cv2.imread(f"./data/thumbnail/{video_id}.jpg", cv2.IMREAD_COLOR)
    b = img.T[0].flatten().mean()
    g = img.T[1].flatten().mean()
    r = img.T[2].flatten().mean()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img_hsv.T[0].flatten().mean()
    s = img_hsv.T[1].flatten().mean()
    v = img_hsv.T[2].flatten().mean()
    return r, g, b, h, s, v


def make_thumbnail_features():
    tqdm.pandas()
    train = pd.read_csv("./data/input/train_data.csv")
    test = pd.read_csv("./data/input/test_data.csv")

    sub = pd.DataFrame()
    sub["video_id"] = train.video_id
    sub[["img_R", "img_G", "img_B", "img_H", "img_S", "img_V"]] = train.progress_apply(
        lambda x: get_grb_hsv(x["video_id"]), axis=1, result_type="expand")
    sub.to_csv("./data/input/train_thumbnail.csv", index=False)

    sub = pd.DataFrame()
    sub["video_id"] = test.video_id
    sub[["img_R", "img_G", "img_B", "img_H", "img_S", "img_V"]] = test.progress_apply(
        lambda x: get_grb_hsv(x["video_id"]), axis=1, result_type="expand")
    sub.to_csv("./data/input/test_thumbnail.csv", index=False)
