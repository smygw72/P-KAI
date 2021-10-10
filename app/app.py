import json
import boto3
import base64
import re
import random

from inference.main import main as inference


def handler(event, context):

    print("******lambda event happended******")

    # print(event)
    # sound_path = event['body']['path']
    # print(sound_path)

    # # base64 decode
    # jsons = base64.b64decode(event['body-json']).split(b'\r\n')
    # print(base64.b64decode(event['body-json'])

    # # get filename
    # filename = repr(jsons[1])
    # filename = re.search(r'filename=".*"', filename).group()
    # filename = filename.replace('filename=', '')
    # filename = filename.replace('"', '')

    # # get audio binary data
    # boundary = jsons[0].replace(b'-', b'')
    # audioBody = jsons[4]
    # for i in range(5, len(jsons)):
    #     if jsons[i].find(boundary) != -1:
    #         break
    #     else:
    #         audioBody = audioBody + b'\r\n' + jsons[i]

    # # s3 update
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket('test-multipart-request')
    # bucket.put_object(
    #     Body=audioBody,
    #     Key=filename
    # )

    # audioBody = None

    # exec model
    sound_path = None
    score = exec_model(sound_path)
    # score = 1

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
