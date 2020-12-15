docker run -p 8501:8501 -p 8500:8500 --mount type=bind,\
source=/home/sun/Code/tensorflow_lr/serving_base64/saved_model_v2,target=/models/flower \
-e MODEL_NAME=flower -t tensorflow/serving:1.14.0 &  