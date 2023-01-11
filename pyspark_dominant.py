from pathlib import Path
from pyspark.sql import SparkSession
import cv2
import numpy as np
from coco_classes import COCO_CLASSES
import json
import os
import sys
import math

# Hostname of our HDFS namenode
HDFS_HOSTNAME = "brick"
# HDFS_HOSTNAME = "g-furnace"

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


if "--cached-pred" in sys.argv:
    colors_df = spark.read.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/results_dominant", header=True, sep=";"
    )
else:
    colors = files.map(get_avg_dominant_color)
    colors_df = colors.toDF(("id", "average_color", "dominant_color"))
    colors_df.write.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/results_dominant",
        mode="overwrite",
        header=True,
        sep=";",
    )


if "--skip1" not in sys.argv:
    colors_df = colors_df.toPandas()
    color_list = colors_df["dominant_color"].to_numpy()
    unique_colors = np.unique(color_list)
    unique_colors = sc.parallelize(unique_colors)

    def count(color):
        images_with_dominant = colors_df.loc[colors_df["dominant_color"] == color]
        return (color, len(images_with_dominant))

    dominant_count = unique_colors.map(lambda c: count(c))
    dominant_count = dominant_count.toDF()

    dominant_count.write.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/results_dominant_count",
        mode="overwrite",
        header=True,
        sep=";",
    )

if "--skip2" not in sys.argv:

    primary_colors = [
        [0, 100, 100],  # red
        [120, 100, 100],  # green
        [240, 100, 100],  # blue
        [180, 100, 100],  # cyan
        [60, 100, 100],  # yellow
        [300, 100, 100],  # magenta
    ]

    primary_count = [0,0,0,0,0,0]

    def c_dist(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    def get_closest_primary(entry):
        file_id = entry[0]
        dominant_c = eval(entry[2])
        
        dist = [c_dist(dominant_c, x) for x in primary_colors]
        minim = min(dist)
        min_index=dist.index(minim)
        return min_index

    colors = colors_df.rdd
    closest = colors.map(get_closest_primary)

    for i in range(0, len(primary_colors)):
        s = closest.filter(lambda x : x == i).count()
        primary_count[i] = s

    Path("./stats/closest_primary").mkdir(parents=True, exist_ok=True)
    for _class in primary_count:
        output = "primary_color;count\n"

        for i in range(0, len(primary_colors)):
            output += f"{primary_colors[i]};{primary_count[i]}\n"

        with open("./stats/closest_primary/results.csv", "w+") as f:
            f.write(output)

        df = sc.parallelize([output]).coalesce(1).toDF(("string"))
        df.write.mode("overwrite").text(f"hdfs://{HDFS_HOSTNAME}:9000/closest_primary")
