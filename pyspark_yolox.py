# Runs yolox-tiny model on images and saves a dictionary
# of every object detected with confidence value exceeding TRESHOLD for every file.

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
import time
import torch
import cv2
import numpy as np
import os
import json
from pyspark.sql import SparkSession

TRESHOLD = 0.4

spark = (
    SparkSession.builder.appName("TEST")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
sc = spark.sparkContext

model_name = "yolox-tiny"
ckpt_file = "yolox_tiny.pth"

exp = get_exp(None, model_name)

exp.test_conf = 0.3
exp.nms = 0.3

model = exp.get_model()
model.eval()

ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])

brmodel = sc.broadcast(model)
brexp = sc.broadcast(exp)

# files = sc.binaryFiles("hdfs://brick:9000/images/*/*/*/*.jpg")
# files = sc.binaryFiles("hdfs://brick:9000/images/0/1/*/*.jpg")
files = sc.binaryFiles("hdfs://brick:9000/images/0/0/0/*0.jpg")

img_labels_df = spark.read.csv(
    "hdfs://brick:9000/metadata/train_labels.csv", header=True
).toPandas()

img_labels_df = img_labels_df.set_index(["id"])

img_labels_bc = sc.broadcast(img_labels_df)


def inference(file):
    expp = brexp.value
    file_name = file[0]
    binary_img = file[1]

    img = cv2.imdecode(np.frombuffer(binary_img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    vt = ValTransform(legacy=False)
    img, _ = vt(img, None, expp.test_size)

    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = brmodel.value.to(device)
    img = img.to(device)

    with torch.no_grad():
        # t0 = time.time()
        # outputs = brmodel.value(img)
        outputs = model(img)
        outputs = postprocess(
            outputs, expp.num_classes, expp.test_conf, expp.nmsthre, class_agnostic=True
        )
        # print("Infer time: {:.4f}s".format(time.time() - t0))
    return (file_name, outputs)


def get_predictions(file):
    file_name, outputs = file
    output = outputs[0]
    file_id = os.path.basename(file_name).split('.')[0]

    if output is None:
        return (file_id, json.dumps({}))

    output = output.cpu()

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    d = {}

    for c, s in zip(cls, scores):
        c = int(c)
        if s > TRESHOLD:
            d[c] = d.get(c, 0) + 1

    dstr = json.dumps(d)
    return (file_id, dstr)


inf = files.map(lambda f: inference(f))
predictions = inf.map(lambda f: get_predictions(f))

predictions_df = predictions.toDF( ("id", "predictions") )
predictions_df.write.csv("hdfs://brick:9000/results_predictions", mode="overwrite", header=True)

predictions_df = predictions_df.toPandas()
predictions_df = predictions_df.set_index(["id"])

labels_to_check = img_labels_bc.value.loc[predictions_df.index , :]
labels_to_check = np.unique ( np.array([val[0] for val in labels_to_check.values]))

def count_objects(class_label):
    df = img_labels_bc.value
    images_of_class = df.loc[df['landmark_id'] == str(class_label)]
    images_of_class = [ index for index in images_of_class.index] 

    d = {}

    file_counter = 0
    for image_id in images_of_class:
        if image_id in predictions_df.index:
            image_dict = predictions_df.at[image_id, 'predictions']
            image_dict = json.loads(image_dict)
            for key in image_dict:
                d[key] = d.get(key, 0) + image_dict[key]
            file_counter+=1

    avgs = {}
    for key in d:
        avgs[key] = d[key] / file_counter
    
    #must cast label to str because numpyt str is not accespted by csv writer
    return (str(class_label), json.dumps(d), json.dumps(avgs))
            

classes = sc.parallelize(labels_to_check)
sums = classes.map(count_objects)
sums_df = sums.toDF(("landmark_id", "predictions", "averages"))
sums_df.write.csv("hdfs://brick:9000/results_predictions_per_class", mode="overwrite", header=True)
