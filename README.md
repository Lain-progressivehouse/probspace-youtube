# YouTube動画視聴回数予測
Public 4th, Private 6th

URL: https://prob.space/competitions/youtube-view-count

## はじめに
マルチモーダルなコンペで工夫のしがいがあり非常に楽しいコンペでした．コンペを開催してくださった運営と，参加者の皆さん，そしてチームを組んで下さったupuraさんに感謝を申し上げます．
以下のSolutionは私の取り組みです．



## 特徴量
- テキストデータは単語や日本語，数字などのカウントを主に使用
- カテゴリ毎に集計したり数値同士を演算したり
- EfficientNetで画像の特徴量抽出をしたものをumapで次元圧縮したが性能の改善に寄与しなかった
- Bertも同様に性能の改善に寄与しなかった

## like, dislike, comment_countの補完
like, dislike, comment_countを予測するモデルを作成し，comments_disabled, ratings_disabledがTrueであるデータを補間しました．学習には補完ありと補完なしの2種類のデータを使用してモデルを学習しました．

## Targetの変更
- 通常のTarget
- categoryIdでTarget Encodingした値とTargetの差分
- Targetを期間(collection_dateとpublishedAtの差分)で除算したもの

の3つをTargetとして学習させました．

## Pseudo Labeling
KFoldで予測したK個のtestの予測値の分散が低いものを精度が高いと仮定して，testの予測の分散が低いもののうち30%を使用してpreudo labelingを行いました．CVは大幅に下がりLBは少しだけ改善しました．

## Validation
StratifiedKFold: train.y.apply(np.log1p) // 2

## Model
LightGBMを使用．
補完ありと補完なしの2種類のデータと3種類のTargetの計6種類のModelを作成し，平均をとりました．

## Ensemble
ベストモデルは以下の4つのモデルの平均
- upuraさんのモデル
- Null Importanceで特徴量選択，Pseudo Labelingありのモデル
- Null Importanceで特徴量選択，Pseudo Labelingなしのモデル
- LightBGMのImportanceの上位で特徴量選択，Pseudo Labelingありのモデル

## Github
https://github.com/Lain-progressivehouse/probspace-youtube
