mkdir -p ~/.aws-lambda-rie && \
        curl -Lo ~/.aws-lambda-rie/aws-lambda-rie \
        https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && \
        chmod +x ~/.aws-lambda-rie/aws-lambda-rie

docker build -f Dockerfile.Lambda -t psa_lambda .
docker run --name psa_lambda --rm -p 9000:8080 \
            -v ~/.aws-lambda-rie:/aws-lambda \
            --entrypoint /aws-lambda/aws-lambda-rie \
            psa \
            /usr/local/bin/python -m awslambdaric app.handler