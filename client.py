'''
@Cute-Yang
'''


import base64
import json
import requests
import time

image_path="daisy.jpg"
url="http://localhost:8501/v1/models/flower:predict"

def predict(image_path,url):
    with open(image_path,"rb") as f:
        image_base64=base64.b64encode(f.read()).decode("utf-8")
    
    headers={
        "content-type":"application/json"
    }

    data=json.dumps(
        {
            "signature_name":"flower_serving",
            "instances":[
                {
                    "b64":image_base64
                }
            ]
        }
    )

    start_time=time.time()
    response=requests.post(url,data,headers=headers)
    end_time=time.time()
    print(response,response.text)

    print("Used time %.4fs"%(end_time-start_time))
if __name__=="__main__":
    predict(image_path,url)