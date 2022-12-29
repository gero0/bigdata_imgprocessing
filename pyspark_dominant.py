from pyspark.sql import SparkSession
import cv2
import numpy as np
from coco_classes import COCO_CLASSES
import json
import os
import sys

# Hostname of our HDFS namenode
# HDFS_HOSTNAME = "brick"
HDFS_HOSTNAME = "g-furnace"

n_colors = 4

# Create spark context
spark = (
    SparkSession.builder.appName("stat-generation")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
sc = spark.sparkContext

files = sc.binaryFiles(f"hdfs://{HDFS_HOSTNAME}:9000/images/*/*/*/*.jpg")
# files = sc.binaryFiles(f"hdfs://{HDFS_HOSTNAME}:9000/images/0/0/0/*.jpg")


def get_avg_dominant_color(file):
    file_name = file[0]
    print("Now processing", file_name)
    file_id = os.path.basename(file_name).split(".")[0]
    binary_img = file[1]

    img = cv2.imdecode(np.frombuffer(binary_img, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    # calculate avg color
    average = img.mean(axis=0).mean(axis=0)

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img.reshape(-1, 3)

    # k-means clustering
    pixels = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    average = json.dumps([int(average[0]), int(average[1]), int(average[2])])
    dominant = json.dumps([int(dominant[0]), int(dominant[1]), int(dominant[2])])

    print(average)
    print(dominant)

    return (file_id, average, dominant)


colors = files.map(get_avg_dominant_color)
colors_df = colors.toDF(("id", "average_color", "dominant_color"))
colors_df.write.csv(
    f"hdfs://{HDFS_HOSTNAME}:9000/results_dominant",
    mode="overwrite",
    header=True,
    sep=";",
)
