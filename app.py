import os
import datetime
import json
import re
import boto3
import subprocess

from inference import main as inference


def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key


def handler(event, context):

    print("******lambda event happened******")

    region = 'ap-northeast-1'
    instances = [os.environ['INSTANCE_ID']]

    ec2 = boto3.client('ec2', region_name=region)
    action = event["Action"]
    response = ec2.describe_instances(InstanceIds=instances)
    ec2_status = response['Reservations'][0]['Instances'][0]['State']['Name']
    print(instances[0] + ' instance is ' + ec2_status + ' now.')

    ec2.start_instances(InstanceIds=instances)
    print('started your instance: ' + str(instances[0]))
    response = ec2_run(event)
    ec2.stop_instances(InstanceIds=instances)

    return response


def ec2_run(event):
    print(event)
    path = event['path']

    s3 = boto3.resource('s3')
    bucket_id, key = split_s3_path(path)
    print("Bucket: " + bucket_id)
    print("Key: " + key)

    time = datetime.datetime.now()
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
    tmp_path = f'/tmp/{timestamp}.mp3'
    print("tmp path: " + tmp_path)

    bucket = s3.Bucket(bucket_id)
    bucket.download_file(key, tmp_path)

    print(subprocess.run(["ls", "-l", "/tmp"], stdout=subprocess.PIPE))

    score = inference(sound_path=tmp_path, local_or_lambda='lambda')
    os.remove(tmp_path)

    # response body
    data = {
        'score': score
    }

    return {
        # 'isBase64Encoded': False,
        'statusCode': 200,
        # 'headers':{},
        'body': json.dumps(data)
    }

