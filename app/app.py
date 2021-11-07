import datetime
import json
import boto3
import subprocess

from inference import main as inference


def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key


def handler(event, context):

    print("******lambda event happended******")

    print(event)
    path = event['path']

    s3 = boto3.resource('s3')
    bucket, key = split_s3_path(path)
    print(bucket)
    print(key)

    time = datetime.datetime.now()
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')

    tmp_path = f'/tmp/{timestamp}.mp3'

    bucket = s3.Bucket(bucket)
    bucket.download_file(key, tmp_path)
    print(subprocess.run(["ls", "-l", "/tmp"], stdout=subprocess.PIPE))

    score = exec_model(tmp_path)

    # responce body
    data = {
        'score': score
    }

    return {
        # 'isBase64Encoded': False,
        'statusCode': 200,
        # 'headers':{},
        'body': json.dumps(data)
    }


def exec_model(sound_path):
    score = inference(sound_path)
    return score
