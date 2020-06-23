from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import tokenizers
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

model = EfficientNet.from_pretrained('efficientnet-b7', advprop=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')
print(f"使用デバイス: {device}")
model.to(device)


class ThumbnailDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tfms = transforms.Compose(
            [
                transforms.Resize((90, 120)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        data['images'] = self.get_image_data(row)

        return data

    def __len__(self):
        return len(self.df)

    def get_image_data(self, row):
        image_tensor = self.tfms(Image.open(f"./data/thumbnail/{row.video_id}.jpg").convert('RGB'))
        return image_tensor


def get_features(df):
    ds = ThumbnailDataset(df)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    for batch in tqdm(dl):
        input_images = batch["images"]
        bs = input_images.size(0)
        output = model.extract_features(input_images)
        output = nn.AdaptiveAvgPool2d(1)(output)
        output = output.view(bs, -1)
        output = output.to(cpu)
        preds.append(output.detach().clone().numpy())
    return np.concatenate(preds, axis=0)


def main():
    train = pd.read_csv("./data/input/train_data.csv")
    test = pd.read_csv("./data/input/test_data.csv")

    train_image_features = get_features(train)
    test_image_features = get_features(test)
    num = train_image_features.shape[1]
    cols = [f"image_{i}" for i in range(num)]
    train_image_df = pd.DataFrame(train_image_features, columns=cols)
    test_image_df = pd.DataFrame(test_image_features, columns=cols)
    train_image_df.to_csv("./data/input/train_image_features.csv", index=False)
    test_image_df.to_csv("./data/input/test_image_features.csv", index=False)
