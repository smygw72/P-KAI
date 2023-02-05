# P-KAI (Piano-Karaoke AI)

## Overview

P-KAI is a 2D-CNN that calculates your piano skill from an audio.
This task is very difficult because the model needs to extract fine-grained features for skill assessment from pairwise annotation, leading that weakly-supervised learning.

## Model

We implemented four kinds of model below.
PDR and APR are 2D-CNN with time-series modeling called [TSN (Temporal Segment Network)](https://arxiv.org/abs/1608.00859).

- PDR (Pairwise Deep Ranking)
- APR (Attention Pairwise Ranking)

## Dataset

We used two kinds of small dataset (size < 100) and a pairwise annotation representing which one is superior/inferior/equivalent for each pair.
One dataset is from [PSA(Piano-Skill-Assessment)](https://github.com/ParitoshParmar/Piano-Skills-Assessment) and the other is our original dataset collected from Youtube.
PSA contains 62 different songs and our dataset 32 same songs (”For Elise” by Beethoven).

## Example

hoge

## Performance

Accuracy on k-folds cross validation (k=3)

- PDR: 84%
- APR: XX%

## Release

Web service would be released [here](https://feature.d3c5bfncikrlyq.amplifyapp.com) (COMING SOON!!).

## How to run

### Google Colaboratory (supports training, inference, and visualization)

See [here](https://colab.research.google.com/drive/1CDboBGtF6i3MOdFJEbY6IBdowrJfEsj_?usp=sharing)

### Docker (supports CPU environment)

　0. (Option) Install Docker.

    bash ./setup/setup_linux.sh

　1. Build image and create container.

    docker build -f ./setup/Dockerfile.CPU -t psa_cpu .

　2. Run container.

    docker container exec psa_cpu sh -c "python inference.py"

### AWS Lambda (supports inference with cpu)

　0. (Option) Install Docker.

    bash ./setup/setup_linux.sh

　1. Build and push image to AWS ECR.

    bash ./setup/deploy.sh

　2. Connect Lambda image to ECR image.

    - select "arm64" architecture.

　3. Sending request to the container.

    curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'

### Manual setup

　1. Create python environment.

    python -m venv venv
    source venv/bin/activate
    pip install -r ./setup/requirements_cpu.txt

　2. Download dataset from Youtube.

    python ./preprocessing/download.py

　3. Model learning (config is defined in "./config/*.yaml").

    python learning.py [--config ./config/{PDR/APR}.yaml]

　4. Model inference (Used model is in "./model/**/state_dict.pt").

    python inference.py

## How to update dataset

　1. Update ./annotation/youtube.csv

　2. Create all_pair.csv file

    python ./preprocessing/make_pair.py

　3. Fill a label column (0/1/-1 instead of 'X') in all_pair.csv. You can use semi-automatic labeling with stdin.

    python ./preprocessing/annotate.py

　4. Split all pairs into train/test and based on k-fold cross validation

    python ./preprocessing/split_pair.py

## Reference

- Architecture
  - PDR: [Who's Better? Who's Best? Pairwise Deep Ranking for Skill Determination (CVPR'18)](https://arxiv.org/abs/1703.09913)
  - APR: [Attention Pairwise Ranking (MIRU'20)](https://github.com/mosa-mprg/attention_pairwise_ranking)
  - TSN: [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)
- Dataset
  - [ParitoshParmar/Piano-Skills-Assessment](https://github.com/ParitoshParmar/Piano-Skills-Assessment)
- Preprocessing
  - [Piano Skills Assessment (IEEE MMSP, 2021)](https://arxiv.org/abs/2101.04884)
  - [Audio Classification using Librosa and Pytorch (blog)](https://medium.com/@hasithsura/audio-classification-d37a82d6715)
