#!/bin/sh
# https://docs.aws.amazon.com/AmazonECR/latest/userguide/registry_auth.html
REGION='ap-northeast-1'
AWS_ACCOUNT_ID='655146334141'
LOCAL_CONTAINER='psa_lambda'
ECR_CONTAINER='pkai-container'

# 1. Build image
# docker-compose -f docker-compose.lambda.yml up -d --build
docker build -f Dockerfile.Lambda -t $LOCAL_CONTAINER .

# 2. tag image
docker tag $LOCAL_CONTAINER $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_CONTAINER

# 3. Authenticate Docker to an Amazon ECR private registry with get-login
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 4. Push image to AWS ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_CONTAINER