# P-KAI (Piano-Karaoke AI)
P-KAI is CNN-based model with metric learning that calculates your piano skill from an audio(.mp3).
Web service is [here](https://feature.d3c5bfncikrlyq.amplifyapp.com).

## How to run

### Google Colaboratory (supports training, inference, and visualization)

See [here](https://colab.research.google.com/drive/1CDboBGtF6i3MOdFJEbY6IBdowrJfEsj_?usp=sharing)

### Docker (supports inference with cpu)

　0. (Option) Install Docker.

    bash setup.sh

　1. Build image and create container.

    docker build -f Dockerfile.CPU -t psa_cpu .

　2. Run container.

    docker container exec psa_cpu sh -c "python inference.py"

### AWS Lambda (supports inference with cpu)

　0. (Option) Install Docker.

    bash setup.sh

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

　4. Learn model.

    python learning.py

　5. Use model.

    python inference.py

## How to update dataset

　1. Update Youtube.csv

　2. Create all_pair.csv file

    python make_pair.py

　3. Fill a label column (0/1/-1 instead of 'X') in all_pair.csv. You can use semi-automatic labeling with stdin.

    python annotate.py

　4. Split all pairs into train/test and based on k-fold cross validation

    python split_pair.py

## Performance note for each version

- 0.1: ResNet50 + marginal loss (acc: X)
- 0.2: XXXXXXXXXXX (acc: X)

## Reference

- Architecture
    - [Who's Better? Who's Best? Pairwise Deep Ranking for Skill Determination (CVPR'18)](https://arxiv.org/abs/1703.09913)
- Dataset
    - [ParitoshParmar/Piano-Skills-Assessment](https://github.com/ParitoshParmar/Piano-Skills-Assessment)
- Preprocessing
    -  [Piano Skills Assessment (arXiv'21)](https://arxiv.org/abs/2101.04884)
    - [Audio Classification using Librosa and Pytorch (blog)](https://medium.com/@hasithsura/audio-classification-d37a82d6715)