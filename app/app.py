import json
import boto3
import base64
import re
import json
import random

def lambda_handler(event, context):
    
    # base64 decode
    jsons = base64.b64decode(event['body-json']).split(b'\r\n')
    # print(base64.b64decode(event['body-json']))
    
    # get filename
    filename = repr(jsons[1])
    filename = re.search(r'filename=".*"', filename).group()
    filename = filename.replace('filename=','')
    filename = filename.replace('"','')

    # get audio binary data
    boundary = jsons[0].replace(b'-', b'')
    audioBody = jsons[4]
    for i in range(5, len(jsons)):
        if jsons[i].find(boundary) != -1:
            break
        else:
            audioBody = audioBody + b'\r\n' + jsons[i]

    # s3 update
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('test-multipart-request')
    bucket.put_object(
        Body = audioBody,
        Key = filename
    )
    
    # exec model
    score = exec_model(audioBody)

    # responce body
    data = {
        'score': score
    }

    return {
        # 'isBase64Encoded': False,
        # 'statusCode': 200,
        # 'headers':{},
        'body': json.dumps(data)
    }

# dummy function for call model
def exec_model(audio):
    
    # dummy score
    score = random.randint(0, 100)
    
    return score
    
    