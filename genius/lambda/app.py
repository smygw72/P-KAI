import json
import boto3
import base64
import re
import json
import random
from inference.main import main

# lambdaから最初に呼び出される関数
# デバック完了したらコメントアウト戻す
def handler(event, context):
    
    # # base64 decode
    # jsons = base64.b64decode(event['body-json']).split(b'\r\n')
    # print(base64.b64decode(event['body-json']))
    
    # # get filename
    # filename = repr(jsons[1])
    # filename = re.search(r'filename=".*"', filename).group()
    # filename = filename.replace('filename=','')
    # filename = filename.replace('"','')

    # # get audio binary data
    # boundary = jsons[0].replace(b'-', b'')
    audioBody = '' # デバック用
    # audioBody = jsons[4]
    # for i in range(5, len(jsons)):
    #     if jsons[i].find(boundary) != -1:
    #         break
    #     else:
    #         audioBody = audioBody + b'\r\n' + jsons[i]
    
    # モデル呼出
    score = exec_model(audioBody)

    # レスポンス除法
    data = {
        'score': score
    }

    return {
        'body': json.dumps(data)
    }

# モデル呼び出し用関数（分ける必要ないかも）
def exec_model(audioBody):
    
    # dummy score
    score = main() # 実際には引数にaudioBodyを渡す
    
    return score
    
    