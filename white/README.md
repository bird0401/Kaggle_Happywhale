# HappyWhale 

## 2022/03/22

### simple ensemble of public best kernels
現状トップのノートブック見ていく
https://www.kaggle.com/code/yamsam/simple-ensemble-of-public-best-kernels
有名どころをアンサンブルしてる。

元になってるノートブックは以下の四つ

- https://www.kaggle.com/aikhmelnytskyy/happywhale-arcface-baseline-eff7-tpu-768-inference

- https://www.kaggle.com/nghiahoangtrung/0-720-eff-b5-640-rotate

- https://www.kaggle.com/aikhmelnytskyy/happywhale-effnet-b7-fork-with-detic-training

- https://www.kaggle.com/andrej0marinchenko/happywhale-0-679




### happywhale_arcface_baseline_eff7_tpu_768_inference (https://www.kaggle.com/code/aikhmelnytskyy/happywhale-arcface-baseline-eff7-tpu-768-inference/notebook)
元になったノートブック　https://www.kaggle.com/ks2019/happywhale-arcface-baseline-tpu

Model : EfficientNetB6

今まで：一つのモデルをone foldで学習させたものだけを使用していた。

今回　：五つのモデルを別々に学習させた。-> **スコア5%上がった**


データの前処理
- 背鰭に合わせて正方形に切り取り


# ArcMarginProductを使う、あとで調べる

Modelの構造
<img width="577" alt="image" src="https://user-images.githubusercontent.com/48637189/159396103-86ac750e-fc2b-49d1-afac-c2f3c2267bd3.png">

学習率はよく見るこの形を使う。

<img width="455" alt="image" src="https://user-images.githubusercontent.com/48637189/159396455-5ba82a5c-fc45-4bc7-bd89-47f261bad203.png">

Better than median を使ってる。
https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/282735


事前に学習した五つのモデルの重みをロードしている。

このノートブックでは学習を回していない。
https://www.kaggle.com/code/aikhmelnytskyy/happywhale-effnet-b7-fork-with-detic-training/notebook
学習はこのURL

閾値をいろいろ試し最適な値を求める。

**Comment**
Q. TFRecordsから既に処理済みのTest Datasetを取得してたけどどうやってるの？

A. Private datasetにすでにアクセスしてる。train の方でかくにんしてね。

Q. ひとつの画像に五つの予測をしてる？

A. 異なるfoldで学習させた五つのモデルの平均値を取得してる。

このノートブックについてのDisscussion
https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/310119



