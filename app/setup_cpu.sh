docker build -f Dockerfile.CPU -t psa_cpu .
docker container run -d --rm --name psa psa_cpu