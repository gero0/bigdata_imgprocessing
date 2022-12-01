# Runs yolox-tiny model on images and saves a dictionary
# of every object detected with confidence value exceeding TRESHOLD for every file.

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
import time
import torch
import cv2
import numpy as np
import os
import sys
import json
from pyspark.sql import SparkSession

#Min treshold for object detection
TRESHOLD = 0.4

#Hostname of our HDFS namenode
HDFS_HOSTNAME= 'brick'
# HDFS_HOSTNAME= 'g-furnace'

#Minimum partitions to split data into, so we don;t have idle threads
MIN_PARTITIONS = 8

#Create spark context
spark = (
    SparkSession.builder.appName("object-detection")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
sc = spark.sparkContext

#Configuring and loading YOLO model and broadcasting it to workers
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

#load file with image_id, landmark_id pairs and broadcast it to workers
img_labels_df = spark.read.csv(
    f"hdfs://{HDFS_HOSTNAME}:9000/metadata/train_labels.csv", header=True
).toPandas()

img_labels_df = img_labels_df.set_index(["id"])

img_labels_bc = sc.broadcast(img_labels_df)

#Function performing detection on the image and returning filename and positions and labels of detected objects
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

#Discards position data, returns number of occurences of objects per class
def get_predictions(file):
    file_name, outputs = file
    output = outputs[0]
    file_id = os.path.basename(file_name).split(".")[0]

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

#If argument cached-pred is passed, predictions are loaded from hdfs csv files computed earlier
#and the first step is skipped
if "--cached-pred" in sys.argv:
    print('Using cached values from results_predictions')
    predictions_df = spark.read.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/results_predictions", header=True, sep=";"
    )
else:
    #Load files, spark makes partitions automatically using directory structure
    print('Running object detection on files...')
    files = sc.binaryFiles(f"hdfs://{HDFS_HOSTNAME}:9000/images/*/*/*/*.jpg")
    # files = sc.binaryFiles(f"hdfs://{HDFS_HOSTNAME}:9000/images/0/1/*/*.jpg")
    # files = sc.binaryFiles(f"hdfs://{HDFS_HOSTNAME}:9000/images/0/0/0/*.jpg")

    #Ensure we get at least 8 partitions to keep CPU cores busy
    if(files.getNumPartitions() < MIN_PARTITIONS):
        print("Repartitioning small dataset...")
        files = files.repartition(MIN_PARTITIONS)
        print('Repartitioning done')

    inf = files.map(lambda f: inference(f))
    predictions = inf.map(lambda f: get_predictions(f))

    #Write results to csv on HDFS
    predictions_df = predictions.toDF(("id", "predictions"))
    predictions_df.write.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/results_predictions", mode="overwrite", header=True, sep=";"
    )

#Make predicitions dataframe
predictions_df = predictions_df.toPandas()
predictions_df = predictions_df.set_index(["id"])

#Check only landscape labels that appeared in predictions_df (speeds up computation for smaller subsets of the dataset)
labels_to_check = img_labels_bc.value.loc[predictions_df.index, :]
labels_to_check = np.unique(np.array([val[0] for val in labels_to_check.values]))

#Checks all images of class_label landmark_id and sums their detections, also computes the average
def count_objects(class_label):
    df = img_labels_bc.value
    images_of_class = df.loc[df["landmark_id"] == str(class_label)]
    images_of_class = [index for index in images_of_class.index]

    d = {}

    file_counter = 0
    for image_id in images_of_class:
        if image_id in predictions_df.index:
            image_dict = predictions_df.at[image_id, "predictions"]
            image_dict = json.loads(image_dict)
            for key in image_dict:
                d[key] = d.get(key, 0) + image_dict[key]
            file_counter += 1

    avgs = {}
    for key in d:
        avgs[key] = d[key] / file_counter

    # must cast label to str because numpyt str is not accespted by csv writer
    return (str(class_label), str(file_counter), json.dumps(d), json.dumps(avgs))

#Sum occurences of objects per landmark_id
classes = sc.parallelize(labels_to_check)
sums = classes.map(count_objects)

#Save results to file
sums_df = sums.toDF(("landmark_id", "image_count", "predictions_sum", "averages"))
sums_df.write.csv(
    f"hdfs://{HDFS_HOSTNAME}:9000/results_predictions_per_class",
    mode="overwrite",
    header=True,
    sep=";",
)
