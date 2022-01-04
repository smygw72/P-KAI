# P-KAI (Piano-Karaoke AI)

## Overview

P-KAI is a 2D-CNN that calculates your piano skill from an audio.
This task is very difficult because the model needs to extract fine-grained features for skill assessment under a weakly supervised learning.

## Model

We implemented four kinds of model below.
PDR and APR are 2D-CNN with time-series modeling called [TSN (Temporal Segment Network)](https://arxiv.org/abs/1608.00859).
However we thought TSN is not sufficient to extract fine-grained features for skill assessment.
Therefore, we considered APR with Temporal Convolutional Networks.

- PDR (Pairwise Deep Ranking)
- APR (Attention Pairwise Ranking)
- APR with TCN (Temporal Convolutional Networks)
  - APR_TCN: TCN after APR
  - TCN_APR: TCN before APR

## Dataset

We used two kinds of small dataset (< 100) and a pairwise annotation representing which one is superior/inferior/equivalent for each pair.
One dataset is from [PSA(Piano-Skill-Assessment)](https://github.com/ParitoshParmar/Piano-Skills-Assessment) and the other is our original dataset collected from Youtube.
PSA contains 62 different songs and our dataset 32 same songs (”For Elise” by Beethoven).

## Release

Web service would be released [here](https://feature.d3c5bfncikrlyq.amplifyapp.com) (COMING SOON!!).

## How to run

### Google Colaboratory (supports training, inference, and visualization)

See [here](https://colab.research.google.com/drive/1CDboBGtF6i3MOdFJEbY6IBdowrJfEsj_?usp=sharing)

### Docker (supports inference with cpu)

　0. (Option) Install Docker.

    bash setup_linux.sh

　1. Build image and create container.

    docker build -f Dockerfile.CPU -t psa_cpu .

　2. Run container.

    docker container exec psa_cpu sh -c "python inference.py"

### AWS Lambda (supports inference with cpu)

　0. (Option) Install Docker.

    bash setup_linux.sh

　1. Build and push image to AWS ECR.

    bash ./deploy.sh

　2. Sending request to the container.

    curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'

### Manual setup

　1. Create python environment.

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements_cpu.txt

　2. Download dataset from Youtube.

    python download.py

　3. Create MFCC images as model inputs.

    python make_mfcc.py

　4. Learn model (config is defined in "./config/*.yaml").

    python learning.py [--config ./config/{PDR/APR/APR_TCN/TCN_APR}.yaml]

　5. Use model (model is in "./model/**/state_dict.pt").

    python inference.py

## How to update dataset

　1. Update Youtube.csv

　2. Create all_pair.csv file

    python make_pair.py

　3. Fill a label column (0/1/-1 instead of 'X') in all_pair.csv. You can use semi-automatic labeling with stdin.

    python annotate.py

　4. Split all pairs into train/test and based on k-fold cross validation

    python split_pair.py

## Reference

- Architecture
  - PDR: [Who's Better? Who's Best? Pairwise Deep Ranking for Skill Determination (CVPR'18)](https://arxiv.org/abs/1703.09913)
  - APR: [Attention Pairwise Ranking (MIRU'20)](https://github.com/mosa-mprg/attention_pairwise_ranking)
  - TCN: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (arXiv'18)](https://github.com/locuslab/TCN)
  - TSN: [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)
- Dataset
  - [ParitoshParmar/Piano-Skills-Assessment](https://github.com/ParitoshParmar/Piano-Skills-Assessment)
- Preprocessing
  - [Piano Skills Assessment (IEEE MMSP, 2021)](https://arxiv.org/abs/2101.04884)
  - [Audio Classification using Librosa and Pytorch (blog)](https://medium.com/@hasithsura/audio-classification-d37a82d6715)