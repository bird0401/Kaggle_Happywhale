# HappyWhale 

## 2022/03/25
### [D] cropped&resized(512x512) dataset using detic https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/305503

Cropped dataset with Detic

cocoa : small box size

detic : bigger box size

**How to create this dataset:**

1. Detect objects in the class of dolphins, whales, and marine life.
2. Select the largest bbox from the detection result and enlarge bbox by 1.2 times. then, crop&resize it.
3. If not found, just resize the image without cropping.

**Problem**
Since the model may predict smaller proposals with higher confidence, some detection results like the one in the image below are included.
If you are interested in false-positive result, please confirm csv file.

https://i.imgur.com/3cAFvvl.png![image](https://user-images.githubusercontent.com/48637189/160092868-c15c8d4e-cc40-4464-85f6-dacfca88c3e2.png)

Dataset link: https://www.kaggle.com/phalanx/whale2-cropped-dataset


**Comment**

Using this method in one kernel

https://www.kaggle.com/code/lextoumbourou/happywhale-effnet-b6-fork-with-detic-crop/notebook

### [D] 🐧 Things to know before starting image preprocessing https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/308026

Overview of Datsets

- imgae_size : so various.
- night view 

https://i.imgur.com/5IWktBo.png![image](https://user-images.githubusercontent.com/48637189/160099035-ffcd6343-5486-49eb-8a5a-42fa1adc64fc.png)


-multiple indivisuals : more than two subjects
https://i.imgur.com/uQgnu1h.png![image](https://user-images.githubusercontent.com/48637189/160098641-0343a836-cdd0-4f40-8b1a-f456c1b7055c.png)

- landscape : Almost of landscape
https://i.imgur.com/Gt9F5nZ.png![image](https://user-images.githubusercontent.com/48637189/160098822-5516dead-b23c-49c5-8ab5-ea1f9e157b90.png)

- image annotations : some images have digital marking 
https://i.imgur.com/tQj1qxf.png![image](https://user-images.githubusercontent.com/48637189/160098895-48c09e41-431a-4722-a710-2abab6900cb5.png)





---

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

---

### [日本語&ENG] HappyWhale effnetv2-m ゆっくり実況 [infer] https://www.kaggle.com/code/pixyz0130/eng-happywhale-effnetv2-m-infer

efficient-v2 を使うときはTPUを使用する。

https://www.kaggle.com/code/aikhmelnytskyy/happywhale-arcface-baseline-eff-net-kfold5-0-652/notebook
このノートブックのefficient-v1をefficient-v2に変更したのみ

---

### HappyWhale ArcFace Baseline (TPU) https://www.kaggle.com/code/ks2019/happywhale-arcface-baseline-tpu

はじめてArcFaceを用いたノートブック。
かなり多くのノートブックの元になっている感じ。

---

### [Pytorch] ArcFace + GeM Pooling Starter https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter

GeM Pooling 

from https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py



ArcFace 


Comments 

Q. What is 'n_accumulate' in CONFIG?

A. This is for gradient accumulation. In this notebook, it is set to 1 so it makes no difference but in cases of using big models, batch size needs to be reduced(eg. 2) which leads to noisy gradients. So in this case we can accumulate the loss for some batches(eg. 4) and then do a backward pass. This increases our effective batch size to 2 * 4 = 8

<img width="522" alt="image" src="https://user-images.githubusercontent.com/48637189/159489407-bb51e510-ba80-4815-a6f3-a58a84eba795.png">

Q. なんでK-Foldで自動のやつを使わないのか？

A. 今回のコンペなら時間的に余裕があるが、普通は余裕がないためmanualで操作したものを使用する。




