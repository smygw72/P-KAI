docker build -f Dockerfile.CPU -t psa_cpu .
docker container run -d --rm --ipc=host --name psa_cpu psa_cpu