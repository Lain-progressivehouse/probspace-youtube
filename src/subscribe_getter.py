import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
from googleapiclient.discovery import build

api_key = "AIzaSyD2mJQwqR8Y2qOwercUoHUPBcNv45HlbZU"
youtube = build("youtube", "v3", developerKey=api_key)


def get_channel_id_list():
    train = pd.read_csv("./data/input/train_data.csv")
    test = pd.read_csv("./data/input/test_data.csv")
    all_df = pd.concat([train.drop(["y"], axis=1), test])
    return all_df.channelId.unique()


def get_channel_subscribes(channel_id):
    try:
        search_response = youtube.channels().list(
            part='statistics',
            id=channel_id,
        ).execute()
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    sleep(0.5)
    if "items" in search_response.keys():
        try:
            info = search_response["items"][0]["statistics"]
            return info["viewCount"], info["commentCount"], info["subscriberCount"], info["hiddenSubscriberCount"], \
                   info[
                       "videoCount"]
        except:
            return np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def main():
    tqdm.pandas()
    # channel_id_list = get_channel_id_list()
    df = pd.read_csv("./data/input/subscribes.csv")
    # channel_id_list = df[df.isna().any(axis=1)].channelId
    # df = pd.DataFrame()
    # df["channelId"] = channel_id_list
    df[['viewCount', 'commentCount', 'subscriberCount', 'hiddenSubscriberCount', 'videoCount']] \
        = df[df.isna().any(axis=1)].progress_apply(lambda x: get_channel_subscribes(x["channelId"]), axis=1, result_type="expand")
    df.to_csv("./data/input/subscribes.csv", index=False)
