docker build -f Dockerfile.CPU -t psa/lambda_docker .
docker run --name lambda --rm -p 9000:8080 \
          -v ~/.aws-lambda-rie:/aws-lambda \
            --entrypoint /aws-lambda/aws-lambda-rie \
              psa/lambda_docker \
                /usr/local/bin/python -m awslambdaric app.handler