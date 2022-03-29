# HappyWhale 

## 2022/03/29

**よく出てくるデータセットについて**
HappyWhaleSplits：individual id を全て0 index に置き変えた。

https://www.kaggle.com/datasets/ks2019/happywhale-splits

TFRecords：不均衡データセットの場合にそのデータセットの偏りが少なくなるようにtrainとtestを分割しているみたい。具体的にはStratifiedKFoldを用いている。(下画像)

https://www.kaggle.com/ks2019/happywhale-tfrecords　https://www.kaggle.com/ks2019/happywhale-tfrecords-v1


処理しているノートブック：https://www.kaggle.com/code/ks2019/happywhale-tfrecords/notebook

![image](https://user-images.githubusercontent.com/48637189/160507155-0c146ae3-37c1-4f5f-945b-07721b778159.png)

分割の手法　https://www.case-k.jp/entry/2021/02/14/155742#:~:text=StratifiedKFold,%E3%81%84%E3%82%8B%E3%81%93%E3%81%A8%E3%81%8C%E3%82%8F%E3%81%8B%E3%82%8A%E3%81%BE%E3%81%99%E3%80%82



---

## 2022/03/28
### 0.720_🐳&🐬EFF_B5_640_Rotate https://www.kaggle.com/code/nghiahoangtrung/0-720-eff-b5-640-rotate/notebook
https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook ベースになったコード

 (-10, 10 ) degree回転することで大きく精度を向上させた　0.67->0.72

```
# Data augmentation function
def data_augment(posting_id, image, label_group, matches):

    ### CUTOUT
    if tf.random.uniform([])>0.5 and config.CUTOUT:
      N_CUTOUT = 6
      for cutouts in range(N_CUTOUT):
        if tf.random.uniform([])>0.5:
           DIM = config.IMAGE_SIZE
           CUTOUT_LENGTH = DIM//8
           x1 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
           x2 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
           filter_ = tf.concat([tf.zeros((x1,CUTOUT_LENGTH)),tf.ones((CUTOUT_LENGTH,CUTOUT_LENGTH)),tf.zeros((DIM-x1-CUTOUT_LENGTH,CUTOUT_LENGTH))],axis=0)
           filter_ = tf.concat([tf.zeros((DIM,x2)),filter_,tf.zeros((DIM,DIM-x2-CUTOUT_LENGTH))],axis=1)
           cutout = tf.reshape(1-filter_,(DIM,DIM,1))
           image = cutout*image

    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
#     degree = random.uniform(-10, 10)
#     image = tfa.image.rotate(image, degree * math.pi / 180)
    
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    return posting_id, image, label_group, matches
```



### Happywhale - Effnet B7 fork with Detic Training https://www.kaggle.com/code/aikhmelnytskyy/happywhale-effnet-b7-fork-with-detic-training/notebook

Backfins ARCFace TPU Effnet https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook
との違いを見ていく

**データのロードが異なる**

前者
```
GCS_PATH = 'gs://kds-2f25b435592b59d6a2e92f82f0316665f6b69e4768d1296342746e85'  # Get GCS Path from kaggle notebook if GCS Path is expired
if not IS_COLAB:
    GCS_DS_PATH=KaggleDatasets().get_gcs_path('happywhale-tfrecords-bb')
    
train_files = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/happywhale-2022-train*.tfrec')))
test_files = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/happywhale-2022-test*.tfrec')))
print(GCS_PATH)
print(len(train_files),len(test_files),count_data_items(train_files),count_data_items(test_files))
```

後者
```
GCS_PATH = 'gs://kds-d916c3252bf3bc5b3500b904f05f51ce57c8df85221d11b7711bcda9'  # Get GCS Path from kaggle notebook if GCS Path is expired
if not IS_COLAB:
    GCS_PATH1 = KaggleDatasets().get_gcs_path('backfintfrecords')
    
train_files = np.sort(np.array(tf.io.gfile.glob(GCS_PATH1 + '/happywhale-2022-train*.tfrec')))
test_files = np.sort(np.array(tf.io.gfile.glob(GCS_PATH1 + '/happywhale-2022-test*.tfrec')))
print(GCS_PATH)
print(len(train_files),len(test_files),count_data_items(train_files),count_data_items(test_files))
```

前者：effnetv1_b7

後者：effnetv1_b5

結論

データセットの違いと使用モデルの差異

### Happywhale[0.679] https://www.kaggle.com/code/andrej0marinchenko/happywhale-0-679/notebook

わからない

### [D]Releasing my Dorsal Fin Dataset & Code https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/310153

製作者は時間なくて参加できないから後悔したデータセット

Comment

Q　なんで尾鰭だけ？体も使った方がいいのでは？

A データにばらつきが出るため

Q "It includes bounding boxes for 4500 whales with a dorsal fin, not including beluge, southern right whale and gray whale."　ってあるけどなんで三種類は採用していない？

A それらは尾鰭が他の者たちと同じようについていないため

これ参照　https://www.kaggle.com/code/kwentar/what-about-species/notebook



**Object Detection**

It includes bounding boxes for 4500 whales with a dorsal fin, not including beluge, southern right whale and gray whale.
Please upvote the dataset if you are using it in your pipeline
Here it is:
https://www.kaggle.com/jpbremer/backfin-annotations



### ⚡️ [EDA] ⚡️ EDA + Visualization + Augmentation 🔥🔥 https://www.kaggle.com/code/sahamed/eda-visualization-augmentation

データ数や提出形式などがかなり詳しく書いてあり、とても参考になる。

```
bottlenose_dolphin           9664
beluga                       7443
humpback_whale               7392
blue_whale                   4830
false_killer_whale           3326
dusky_dolphin                3139
spinner_dolphin              1700
melon_headed_whale           1689
minke_whale                  1608
killer_whale                 1493
fin_whale                    1324
gray_whale                   1123
bottlenose_dolpin            1117
kiler_whale                   962
southern_right_whale          866
spotted_dolphin               490
sei_whale                     428
short_finned_pilot_whale      367
common_dolphin                347
cuviers_beaked_whale          341
pilot_whale                   262
long_finned_pilot_whale       238
white_sided_dolphin           229
brydes_whale                  154
pantropic_spotted_dolphin     145
globis                        116
commersons_dolphin             90
pygmy_killer_whale             76
rough_toothed_dolphin          60
frasiers_dolphin               14
Name: species, dtype: int64
```

種族ごとの偏りはかなりある。

**Data Cleaning**

Fixing Duplicate Labels
- bottlenose_dolpin -> bottlenose_dolphin
- kiler_whale -> killer_whale
- beluga -> beluga_whale


Changing Label due to extreme similarities


- globis & pilot_whale -> short_finned_pilot_whale


**Missing data**

None

**Top 5 least frequent individual**

画像が一枚しかない個体は画像を複製する。

<img width="659" alt="image" src="https://user-images.githubusercontent.com/48637189/160413422-bd7178ab-2a57-4795-8e0d-0bab3136461a.png">


画像が5枚以下の個体は40%もいる




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

- image duplicates : some images are same
- lighting : sometimes the water can be greenish, sometimes bluish, sometimes pink (because of a sunset for example). 
- some onject : penguins and people, board
- ice : many pictures contain ice 

**Comments**

In some cases it is better to discard some images!

**Conclusion**

- Change RGB into B&W
- Can we use aditional data (Ice, Penguin, People)

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




