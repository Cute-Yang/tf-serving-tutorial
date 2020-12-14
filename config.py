# IMAGE_DIR = "/home/yang/图片/CNN数据集/flower_photos/"
'''
all the params in training
'''
IMAGE_SIZE=224
TRAIN_FILE_NUMS=3000
NUM_CLASSES = 5
DEST_DIR = "data"
TRAIN_DATASET_RATE = 0.8
VALIDATION_DATASET_RATE = 0.2
TEST_DATASET_RATE = 0.1
LEARNING_RATE = 1e-4
EPOCHES = 40
BATCH_SIZE = 8
BATCHES=TRAIN_FILE_NUMS//BATCH_SIZE
DECAY_STEPS=500
DECAY_RATE=0.96

ENCODE_MAP = {"sunflowers": 0, "roses": 1,"tulips": 2, "daisy": 3, "dandelion": 4}
DECODE_MAP = {0: "sunflowers", 1: "roses",2: "tulips", 3: "daisy", 4: "dandelion"}
DECODE_MAP_CN={0:"向日葵",1:"玫瑰花",2:"郁金香",3:"雏菊",4:"蒲公英"}
