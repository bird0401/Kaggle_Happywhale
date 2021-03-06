
# 3月
やったdiscussion
- https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/305503
## 21
https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/305503  
- Deticによる切り抜きデータセットを作成
- dataset link: https://www.kaggle.com/phalanx/whale2-cropped-dataset
- Detic：imagenet-2iKをweakly-superviedで学習させたもの。
- cocoより広い意味領域での検出が可能になるので今回の問題に最適
- cocoのような検出データセットでは、一般的な検出方法でモデルを学習する。一方、imagenet-21Kでは、最大のrpn-proposalを選択し、分類損失を計算する。
- RPN：物体が写っている場所と、その矩形の形を検出する機械学習モデル

<img src="https://user-images.githubusercontent.com/53257509/159230487-ecb3910e-fa1e-488c-be10-f1888a3536ae.png" width=50%>

このデータセットの作り方
1. イルカ、クジラ、海洋生物のクラスで物体を検出する。
2. 検出結果から最大のbboxを選択し、bboxを1.2倍に拡大した後、クロップ＆リサイズする。
3. 検出されなかった場合は、クロップせずにそのままリサイズする。

問題点
- モデルはより小さな領域を高い信頼度で予測することもある。
- そのため下画像のような検出結果も含まれる。

<img src="https://user-images.githubusercontent.com/53257509/159230291-c9a5beed-5118-4bba-b1a9-3d130f704f48.png" width=25%>

## 23
- Detic使ってみた in colab: https://colab.research.google.com/drive/1lp5NbqiC_YIsWTHAGeWbKhb0DJ8x6Or0
### detic
- imagenetを使った事前学習モデル
- 画像を入れると、そこに存在するオブジェクトを矩形で囲い、名詞と確率のペアを出力
- 20000クラスの予測ができる
- 正解ラベルを絞ることも可能

<img src="https://user-images.githubusercontent.com/53257509/159693326-443a167c-dba2-44fe-9d80-76b78a469256.JPG" width=40%> <img src="https://user-images.githubusercontent.com/53257509/159693338-7c87482c-3719-4fec-bd97-22b399e5e3ff.jpg" width=40%>

### 次やること
- これを利用した予測モデル: https://www.kaggle.com/code/dragonzhang/happywhale-effnet-b7-fork-with-detic-crop

## 27
### ベースラインモデルの確認
https://colab.research.google.com/drive/1qlhjcU584XILKdXaUQoqEv1IiD3xW9zW  にコメントアウトしてまとめた。datasetをcolabで用いる方法がわからないため止まっている。
### notebookの使い方確認
- GPU、TPU１週間あたり30時間まで
- TPUは、colabと比較してKaggleの方が早い
- Tensorflowを使う場合はGPUよりTPUの方が良い
- link to githubすればnotebookをsaveした段階で、gitにcommitできる

## 31
### ベースラインモデルの確認
https://colab.research.google.com/drive/1qlhjcU584XILKdXaUQoqEv1IiD3xW9zW  にコメントアウトしてまとめた。

### Kaggle APIの導入
参考サイト：https://dreamer-uma.com/kaggle-api-colab/
途中までやった

# 4月
## 1
### 引き続きKaggle APIの導入
- 違う参考サイトみてその通りにやったら、ちゃんと設定できた。
- 参考サイト:https://kopaprin.hatenadiary.jp/entry/2020/05/07/083000
- ダウンロードに５分くらいかかるので、ドライブにダウンロードして、毎回そこからマウントした方が早い
- `!kaggle competitions download -c happy-whale-and-dolphin -p /content/drive/MyDrive/Kaggle/HappyWhale-2022`のようにダウンロードできる

## 2
- データセットのドライブ上でのunzipに時間かかりすぎるのと、データ量が多くcolabのディスク容量を突破してしまう
- 拡張機能のzip extracterをインストール。colab上ではなく、ドライブ上でunzipしようとした。

## 4
- zip extracterではエラーになったっぽい。ファイルサイズ大きすぎが原因と思われる。
- ローカルにダウンロードすると確実にクラッシュする。数百GBあるので。
- やはりドライブ上でunzipするしかなさそう。時間すごいかかるから、ランタイム切れたらリセットされないようにすぐ対応する必要あり。
- ドライブでunzipするとディスク容量足りなくなって途中で終わる
- proにしようと思ったけど、大学のアカウントだと有料版に変更することができない。
- kaggle内で作業するしかなさそう
- notebook使ったらすぐ使えなくなった

## 5
- ベースラインのsubmitした
- [前回のクジラコンペの解法](https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/304504)みた
- 前回のクジラコンペ
  - 212位以内が銅圏内
  - ルールは今回と同様  
- [このissue](https://github.com/bird0401/Kaggle_Happywhale/issues/31)にまとめた
- 1位から3位まで
- 全体的にデータセットに対するアプローチは少なめで、モデルの予測手法周りの工夫が多かった。３年前のコンペで、色々と技術的環境が違うので鵜呑みにはしないほうが良い。例えば、現在はefficientNetが主流だが、この時は色々なモデルが乱立していた。実際にresnetやsnenetなど色々なモデルが用いられている。

## 6
- 引き続き前回クジラコンペの上位解法を参照した

## 7
- [現時点で最高精度のnotebook確認した](https://www.kaggle.com/code/nghiahoangtrung/swin-tranform-submission)
  - swinというモデルを使ってる
  - [2021年6月時点で、swin transformがくそ強いらしい](https://www.slideshare.net/ren4yu/swin-transformer-iccv21-best-paper)
  - swinの長所としてdata augmentationをあまりしなくても安定して精度が出る
  - データセット：オリジナル＋private+
- [現時点で2番目の精度のnotebook確認した](https://www.kaggle.com/code/gtownfoster/effv2-l-backfin-embeddings-ensemble/notebook)
  - コード見てると前やったベースラインと結構コード似てる。
  - 5つのEfficientNetB3でアンサンブル
  - EfficientNetV2を用いている
    - [EfficientNetV2に関する記事](https://qiita.com/omiita/items/1d96eae2b15e49235110)
    - EfficientNetV2がすごい強いらしい
    - 2021年4月ごろにgoogleから出てSOTAとなっている
    - 学習時間はeffnetB7よりもかかる。SからXLまである。
  - efficientnetB4-B7
- [arcfaceに関する記事](https://yaakublog.com/deep_metric_learning)
    - 学習時間はeffnetB7よりもかかる。SからXLまである。
    - 学習時間はeffnetB7よりもかかる。SからXLまである。
- [現時点で3番目の精度のnotebook確認した](https://www.kaggle.com/code/nealart/simple-ensemble-of-public-best-kernels-v-2-2-0)
  - notebookにあるベースライン４つに重みをつけてアンサンブルさせただけだが、精度が良くなった
- simple ensamble周りのnotebookはsubmitのスコアごとに重みづけをしているだけ。あまり参考にはならない。最後にやれば良い。
- それらを省くとswinの次に強いのがやはり、[efficientnetB7を使ったモデルになる](https://www.kaggle.com/code/aikhmelnytskyy/happywhale-arcface-baseline-eff7-tpu-768-inference)
- https://www.kaggle.com/code/nghiahoangtrung/0-720-eff-b5-640-rotate
  - https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook に対して、horizontal flip image and random rotate image (-10, 10 ) degreeをしたら精度がだいぶ良くなった。
  - 前に行った左右非対称性を使って、flip後にnew individualにする工夫をすると精度がさらに上がるかもしれない。
- inference周りの処理どのnotebookも似たようなことしている
- 前に行った左右非対称性を使って、flip後にnew individualにする工夫をすると精度がさらに上がるかもしれない。
- 解説notebookを見る限りあまり詳細に解説されていない。他のnotebookも基本コピペでやっているので、swinに関してもそれで良い気がする
- KNNは距離を測るために利用する。

## 8
### swinにinference機能付け足しを行なった
- swinを投稿する前のnotebookをまずは動かした。[ベースとなるnotebook](https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook)が紹介されていため。
- ある程度補完できたのでsubmitしといた。おそらくどこかでエラー起こると思う。
  - inference周りに限らず学習部分など他にも修正箇所があったので補完した。

## 9
### notebookの時系列順に最後まで実行できるか確かめた
- [こちらのissue参照](https://github.com/bird0401/Kaggle_Happywhale/issues/45)
- 最終的にswinが実行できるところまで持って来れた
- 軽く実行できるように、image sizeを小さくしたりしているので、元のnotebookと同じ条件でsave allしてみる。
- TPU利用時間以内に学習が終わるかが懸念点。

## 10
### swin実行時のエラー解消
- [こちらのissue参照](https://github.com/bird0401/Kaggle_Happywhale/issues/45)
- 色々試行錯誤したけど直らなかった
- notebookのcommentのところで同じエラーになっている人を見つけたので、質問をしてその答え待ち。

## 11
### swin実行時のエラー解消
- [こちらのissue参照](https://github.com/bird0401/Kaggle_Happywhale/issues/45)
- 質問の答え返ってきたので、それ参考にしたらエラー全て治った。
- input_size, epoch, model_name元に戻してsave and runした

## 12
- swinは精度の再現ができなかった
- ひとまずpublicのemsembleを提出した

## 14
- segmentationして物体部分だけ白くしたアルファラベルを用いて、4 channel化をすることに取り組んだ。
- whiteのセグメンテーションしたデータをそのまま利用するのが簡単そう。
- 過去クジラの1stの回答ではmaskしたデータをrleで圧縮したものを(id,rle)のペアでcsvに保存していたが、今回その手法を用いる必要があるのか。

## 15
- マージした
- マージ後はslackで作業
