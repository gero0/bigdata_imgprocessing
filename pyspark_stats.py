from pyspark.sql import SparkSession
from string import ascii_uppercase
from pathlib import Path
import json
import os

#Hostname of our HDFS namenode
HDFS_HOSTNAME = "brick"
# HDFS_HOSTNAME = "g-furnace"

#Create spark context
spark = (
    SparkSession.builder.appName("stat-generation")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
sc = spark.sparkContext

#Read detections per landmark_id from HDFS
objects_per_class = spark.read.csv(
    f"hdfs://{HDFS_HOSTNAME}:9000/results_predictions_per_class", header=True, sep=";"
).rdd

#Read file with mappings of landmark_id to landmark name
landmark_names = (
    spark.read.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/metadata/train_label_to_name.csv",
        header=True,
        sep=";",
    )
    .toPandas()
    .set_index("landmark_id")
)

#Stat 1: Sum detections of selected object classes for landmarks that start with given letter
alphabet = [*ascii_uppercase]
classes_of_interest = [0, 1, 2, 11, 13]
detections_per_letter = {}

#return number of detected objects of class _class, but count them only if landmark's name starts with letter
def count_class_for_letter(entry, letter, _class):
    id = entry["landmark_id"]
    name = landmark_names.at[id, "name"]

    if name[0] != letter:
        return 0

    jsonstr = entry["predictions_sum"]
    detections = json.loads(jsonstr)

    try:
        obj_cnt = detections[str(_class)]
    except:
        obj_cnt = 0

    return obj_cnt

#Stat 1: Sum detections of selected object classes for landmarks that start with given letter
for _class in classes_of_interest:
    detections_per_letter[_class] = {}

    for letter in alphabet:
        occurences = objects_per_class.map(
            lambda entry: count_class_for_letter(entry, letter, _class)
        )
        _sum = occurences.sum()
        detections_per_letter[_class][letter] = _sum

#Write results to local FS and HDFS
Path("./stats/alphabet_count").mkdir(parents=True, exist_ok=True)
for _class in detections_per_letter:

    #write to local fs
    output = "letter;count\n"
    for letter in detections_per_letter[_class]:
        output += f"{letter};{detections_per_letter[_class][letter]}\n"

    with open(os.path.join("./stats/alphabet_count", f"{_class}.csv"), "w+") as f:
        f.write(output)

    #why is there no easy way to write a string to a HDFS file in spark?
    df = sc.parallelize([output]).coalesce(1).toDF(("string"))
    df.write.mode('overwrite').text(f"hdfs://{HDFS_HOSTNAME}:9000/alphabet_count/{_class}")

    
    
