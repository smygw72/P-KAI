import json
import boto3
import base64
import re

def lambda_handler(event, context):
    
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('test-multipart-request')

    jsons = base64.b64decode(event['body-json']).split(b'\r\n')
    
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

    bucket.put_object(
        Body = audioBody,
        Key = filename
    )

    return {
        'isBase64Encoded': False,
        'statusCode': 200,
        'headers':{},
        'body': '{"score": 100 }'
    }