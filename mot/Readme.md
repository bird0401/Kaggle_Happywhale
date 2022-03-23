
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

![uQjAQ6i](https://user-images.githubusercontent.com/53257509/159230487-ecb3910e-fa1e-488c-be10-f1888a3536ae.png)

このデータセットの作り方
1. イルカ、クジラ、海洋生物のクラスで物体を検出する。
2. 検出結果から最大のbboxを選択し、bboxを1.2倍に拡大した後、クロップ＆リサイズする。
3. 検出されなかった場合は、クロップせずにそのままリサイズする。

問題点
- モデルはより小さな領域を高い信頼度で予測することもある。
- そのため下画像のような検出結果も含まれる。

![3cAFvvl](https://user-images.githubusercontent.com/53257509/159230291-c9a5beed-5118-4bba-b1a9-3d130f704f48.png)

## 23
- Detic使ってみた in colab: https://colab.research.google.com/drive/1lp5NbqiC_YIsWTHAGeWbKhb0DJ8x6Or0
### detic
- imagenetを使った事前学習モデル
- 画像を入れると、そこに存在するオブジェクトを矩形で囲い、名詞と確率のペアを出力
- 20000クラスの予測ができる
- 正解ラベルを絞ることも可能
### 次やること
- これを利用した予測モデル: https://www.kaggle.com/code/dragonzhang/happywhale-effnet-b7-fork-with-detic-crop
