from pyspark.sql import SparkSession
from string import ascii_uppercase
from pathlib import Path
from coco_classes import COCO_CLASSES
import json
import os
import sys

# Hostname of our HDFS namenode
HDFS_HOSTNAME = "brick"
# HDFS_HOSTNAME = "g-furnace"

# Create spark context
spark = (
    SparkSession.builder.appName("stat-generation")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
sc = spark.sparkContext

# Read detections per landmark_id from HDFS
objects_per_class = spark.read.csv(
    f"hdfs://{HDFS_HOSTNAME}:9000/results_predictions_per_class", header=True, sep=";"
).rdd

# Read file with mappings of landmark_id to landmark name
landmark_names = (
    spark.read.csv(
        f"hdfs://{HDFS_HOSTNAME}:9000/metadata/train_label_to_name.csv",
        header=True,
        sep=";",
    )
    .toPandas()
    .set_index("landmark_id")
)

landmark_names = sc.broadcast(landmark_names)

# Write results to local FS and HDFS
def write_results(dir, headers, dict):
    Path(f"./stats/{dir}").mkdir(parents=True, exist_ok=True)
    for _class in dict:
        output = ";".join(headers) + "\n"

        for obj in dict[_class]:
            output += f"{obj};{dict[_class][obj]}\n"

        with open(os.path.join(f"./stats/{dir}", f"{_class}.csv"), "w+") as f:
            f.write(output)

        df = sc.parallelize([output]).coalesce(1).toDF(("string"))
        df.write.mode("overwrite").text(f"hdfs://{HDFS_HOSTNAME}:9000/{dir}/{_class}")


# return number of detected objects of class _class, but count them only if landmark's name starts with letter
def count_class(entry, _class):
    id = entry["landmark_id"]
    detections = json.loads(entry["predictions_sum"])
    return detections.get(str(_class), 0)


def count_files(entry):
    id = entry["landmark_id"]
    return int(entry["image_count"])


# Stat 1: Sum detections of selected object classes for landmarks that start with given letter
alphabet = [*ascii_uppercase]
classes_of_interest = [0, 1, 2, 11, 13]
# classes_of_interest = [*range(0, len(COCO_CLASSES))]


if "--skip1" not in sys.argv:
    # Stat 1: Sum detections of selected object classes for landmarks that start with given letter
    detections_per_letter = {}
    detections_per_letter_avg = {}

    for _class in classes_of_interest:
        detections_per_letter[_class] = {}
        detections_per_letter_avg[_class] = {}

    for letter in alphabet:
        objects = objects_per_class.filter(
            lambda entry: landmark_names.value.at[entry["landmark_id"], "name"][0] == letter
        )
        images_n = objects.map(lambda entry: count_files(entry)).sum()

        for _class in classes_of_interest:
            occurences = objects.map(lambda entry: count_class(entry, _class)).sum()
            detections_per_letter[_class][letter] = occurences
            try:
                detections_per_letter_avg[_class][letter] = occurences / images_n
            except ZeroDivisionError:
                detections_per_letter_avg[_class][letter] = 0

    write_results("alphabet_count", ["letter", "count"], detections_per_letter)
    write_results(
        "alphabet_count_avg", ["letter", "avg_count"], detections_per_letter_avg
    )

if "--skip2" not in sys.argv:
    cities = ["New York", "Los Angeles", "Detroit", "Paris", "Berlin", "Warsaw"]

    detections_per_city_avg = {}
    for _class in classes_of_interest:
        detections_per_city_avg[_class] = {}

    for city in cities:
        objects = objects_per_class.filter(
            lambda entry: city in landmark_names.value.at[entry["landmark_id"], "name"]
        )
        file_count = objects.map(lambda entry: count_files(entry)).sum()

        for _class in classes_of_interest:
            detections = objects.map(lambda entry: count_class(entry, _class)).sum()
            try:
                detections_per_city_avg[_class][city] = detections / file_count
            except ZeroDivisionError:
                detections_per_city_avg[_class][city] = 0

    write_results(
        "avg_obj_per_city", ["city", "avg_detections"], detections_per_city_avg
    )

if "--skip3" not in sys.argv:
    total_people = objects_per_class.map(lambda entry: count_class(entry, 0)).sum()
    total_files = objects_per_class.map(lambda entry: count_files(entry)).sum()

    people_classes = objects_per_class.filter(
        lambda entry: "people"
        in landmark_names.value.at[entry["landmark_id"], "name"].lower()
    )

    total_people_in_people = people_classes.map(
        lambda entry: count_class(entry, 0)
    ).sum()
    total_people_files = people_classes.map(lambda entry: count_files(entry)).sum()

    try:
        people_avg = total_people / total_files
    except ZeroDivisionError:
        people_avg = 0

    try:
        people_avg_people = total_people_in_people / total_people_files
    except ZeroDivisionError:
        people_avg_people = 0

    people = {0: {"avg_all": people_avg, "avg_people_places": people_avg_people}}

    write_results(
        "people_in_places_with_people", ["files considered", "avg_detections"], people
    )


if "--skip4" not in sys.argv:

    objects_under10 = objects_per_class.filter(
        lambda entry: len(landmark_names.value.at[entry["landmark_id"], "name"]) < 10
    )
    objects_between = objects_per_class.filter(
        lambda entry: len(landmark_names.value.at[entry["landmark_id"], "name"]) >= 10
        and len(landmark_names.value.at[entry["landmark_id"], "name"]) <= 20
    )
    objects_over20 = objects_per_class.filter(
        lambda entry: len(landmark_names.value.at[entry["landmark_id"], "name"]) > 20
    )

    dogs_under10 = objects_under10.map(lambda entry: count_class(entry, 16)).sum()
    files_under10 = objects_under10.map(lambda entry: count_files(entry)).sum()
    try:
        under10_avg = dogs_under10 / files_under10
    except ZeroDivisionError:
        under10_avg = 0

    dogs_between = objects_between.map(lambda entry: count_class(entry, 16)).sum()
    files_between = objects_between.map(lambda entry: count_files(entry)).sum()
    try:
        between_avg = dogs_between / files_between
    except ZeroDivisionError:
        between_avg = 0

    dogs_over20 = objects_over20.map(lambda entry: count_class(entry, 16)).sum()
    files_over20 = objects_over20.map(lambda entry: count_files(entry)).sum()
    try:
        over20_avg = dogs_over20 / files_over20
    except ZeroDivisionError:
        over20_avg = 0

    dogs = {
        16: {
            "under_10_chars": under10_avg,
            "between_10_and_20_chars": between_avg,
            "over_20_chars": over20_avg,
        }
    }

    write_results(
        "dogs_by_name_length", ["length_of_landmark_name", "avg_detections"], dogs
    )
