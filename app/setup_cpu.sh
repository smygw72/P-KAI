docker build -f Dockerfile.CPU -t psa_cpu .
docker container run -d --rm --ipc=host --name psa_cpu psa_cpu
docker container exec psa_cpu bash -c "python /app/inference/main.py"