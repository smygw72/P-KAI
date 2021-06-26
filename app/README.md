# How to run
## Setup with Colaboratory
See [here](https://colab.research.google.com/drive/1CDboBGtF6i3MOdFJEbY6IBdowrJfEsj_?usp=sharing)

## Setup with Docker
0. (Option) Install Docker.
```
bash setup.sh
```
1. Build image and create container.
```
(for any env.) docker build -f Dockerfile.CPU -t psa_cpu .
(for AWS Lambda) docker-compose -f docker-compose.lambda.yml up -d --build
```
2. Run container.
```
(for any env.) docker container exec psa_cpu sh -c "python /app/inference/main.py"
(for AWS Lambda) curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"dummy": "a"}'
```

## Manual setup
1. Make python environment.
```
python -m venv venv
source venv/bin/activate
pip install -r ./app/requirements_cpu.txt
```
2. Download dataset from Youtube.
```
python ./app/utils/download.py
```
3. Make MFCC image as model input.
```
python ./app/utils/make_mfcc.py
```
4. Learn model. (todo)
```
python ./app/learning/main.py
```
5. Use model. (todo)
```
python ./app/inference/main.py
```
# How to update dataset
1. Update Youtube.csv
2. Make all_pair.csv file
```
python ./app/utils/make_pair.py
```
3. Fill a label column (0/1/-1 instead of 'X') in all_pair.csv. You can use semi-automatic labeling with stdin.
```
python ./app/utils/annotate.py
```
4. Make train/test split file based on k-fold cross validation
```
python ./app/utils/split_pair.py
```

# Task
## Completed
- ダウンロード
- トリミング
- アノテーション用ファイル(all_pair.csv)の作成
- アノテーション(3パターン)
    - id1 > id2: 1
    - id1 < id2: -1
    - id1 = id2: 0
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
- 本番環境(AWS lambda)へのデプロイ
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