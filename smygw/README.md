# How to use.
1. Make python environment.
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install ./smygw/requirements.txt
```
2. Download dataset from Youtube.
```
$ python ./smygw/utils/download.py
```
3. Make MFCC image as model input.
```
$ python ./smygw/utils/make_mfcc.py
```
4. Learn model. (todo)
```
$ python ./smygw/learn/main.py
```
5. Use model. (todo)
```
$ python ./smygw/use_model.py
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

## Todo
### must
- アノテーション(3パターン)
    - id1 > id2: 1
    - id1 < id2: -1
    - id1 = id2: 0
- モデル実装
- モデル学習・評価
- 本番環境移行(AWS SageMaker?)
### option
- 可視化(スキルの変動など)
- 特徴量追加([参考](https://qiita.com/__Attsun__/items/e033d689c336315435b3))
- モデルのバイアス除去
    - 雑音除去(or 音源分離)
- 曲追加
- 曲に依存しないモデル学習
- 分散学習
- 楽譜を用いたアライメント
    - 難しい箇所と簡単な箇所で条件付けするため
- ハイパラ最適化

## 検討すべきこと
- MFCCのwindow幅はいくつにすべきか？