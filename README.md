# Kaggle_Happywhale

## train data のcolumns について

image : 画像ファイル名

species : 種の名前

individual_id : 個体ID（これを予測する）

## 提出の形式

一枚の画像に対して、五つの予測をする。

その予測にも１位から５位までの順位付けを行う。

もし、trainの中にない個体である時はnew_individualとする。

## 評価方法

以下のMAP@5を用いる

<img width="306" alt="image" src="https://user-images.githubusercontent.com/48637189/160411405-03183d12-7bc4-42be-ae34-88089991b90a.png">

Uは全体の画像数、P(k)はkの逆数、rel(k)はkが予測に入っていた場合に１を返す。

<img width="400" alt="image" src="https://user-images.githubusercontent.com/48637189/160411713-cb81158f-5ece-4cf7-91e6-a0fefbfda5dd.png">
