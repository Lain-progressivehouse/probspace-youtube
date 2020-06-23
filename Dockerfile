# kaggleのpython環境をベースにする
#FROM gcr.io/kaggle-images/python:v56
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip
RUN apt-get -y install nano wget curl

# ta-libのインストール

# ライブラリの追加インストール
RUN pip3 install -U pip
RUN pip3 install jupyter click numpy matplotlib seaborn pandas tqdm scikit-learn && \
    pip3 install Keras torch torchtext torchvision transformers xgboost lightgbm catboost pykalman feather-format && \
    pip3 install fastprogress japanize-matplotlib && \
    pip3 install logzero && \
    pip3 install pytorch-lightning transformers