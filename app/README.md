# How to run
## Setup with Colaboratory
See [here](https://colab.research.google.com/drive/1CDboBGtF6i3MOdFJEbY6IBdowrJfEsj_?usp=sharing)
## Setup with Docker
1. Build Image.
```
$ docker image build -f ./app/Dockerfile.{GPU/CPU} -t psa ./
```
2. Run container.
```
$ docker container run -d --rm psa --name psa_app
```
3. Execute CMD in the container (Example is as follows).
```
$ docker container exec psa_app sh -c "python /app/inference/main.py"
```

## Manual setup
1. Make python environment.
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r ./app/requirements_{cpu/gpu}.txt
```
2. Download dataset from Youtube.
```
$ python ./app/utils/download.py
```
3. Make MFCC image as model input.
```
$ python ./app/utils/make_mfcc.py
```
4. Learn model. (todo)
```
$ python ./app/learning/main.py
```
5. Use model. (todo)
```
$ python ./app/inference/main.py
```
# How to update dataset
1. Update Youtube.csv
2. Make all_pair.csv file
```
$ python ./app/utils/make_pair.py
```
3. Fill a label column (0/1/-1 instead of 'X') in all_pair.csv. You can use semi-automatic labeling with stdin.
```
$ python ./app/utils/annotate.py
```
4. Make train/test split file based on k-fold cross validation
```
$ python ./app/utils/split_pair.py
```

# Task
## Completed
- ダウンロード
- トリミング
- アノテーション用ファイル(all_pair.csv)の作成
- 学習/テストのデータ分割
- 前処理
    - [Piano Skills Assessment (arXiv'21)](https://arxiv.org/abs/2101.04884)
    - [Audio Classification using Librosa and Pytorch (blog)](https://medium.com/@hasithsura/audio-classification-d37a82d6715)
- モデル設計
    - [Who's Better? Who's Best? Pairwise Deep Ranking for Skill Determination (CVPR'18)](https://arxiv.org/abs/1703.09913)
- モデル実装
- モデル学習・評価
- 可視化(スキルの変動など)

## Todo
### must
- アノテーション(3パターン)
    - id1 > id2: 1
    - id1 < id2: -1
    - id1 = id2: 0
- 本番環境移行
### option
- データ拡張
    - [SpecAugmentation](https://arxiv.org/pdf/1904.08779.pdf)
- 特徴量追加([参考](https://qiita.com/__Attsun__/items/e033d689c336315435b3))
- モデルのバイアス除去
    - 雑音除去(or 音源分離)
- 曲追加
- 曲に依存しないモデル学習
- 分散学習
- 楽譜を用いたアライメント
    - 難しい箇所と簡単な箇所で条件付けするため
- ハイパラ最適化
- midiファイルに変換して学習
- 論理距離を考慮した損失関数設計
- Bokeh使ってインタラクティブにアノテーション

## 検討すべきこと
- MFCCのwindow幅はいくつにすべきか？

## version
- 0.1: ResNet + marginal loss (acc: X)
- 0.2: XXXXXXXXXXX (acc: X)