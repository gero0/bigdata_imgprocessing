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


# Stat 1: Sum detections of selected object classes for landmarks that start with given letter
alphabet = [*ascii_uppercase]
classes_of_interest = [0, 1, 2, 11, 13]
classes_of_interest = [*range(0, len(COCO_CLASSES))]


if "--skip1" not in sys.argv:
    # return number of detected objects of class _class, but count them only if landmark's name starts with letter
    def count_class_for_letter(entry, letter, _class):
        id = entry["landmark_id"]
        name = landmark_names.at[id, "name"]

        if name[0] != letter:
            return 0

        detections = json.loads(entry["predictions_sum"])
        return detections.get(str(_class), 0)

    def count_images_for_letter(entry, letter):
        id = entry["landmark_id"]
        name = landmark_names.at[id, "name"]

        if name[0] != letter:
            return 0

        return int(entry["image_count"])

    # Stat 1: Sum detections of selected object classes for landmarks that start with given letter
    detections_per_letter = {}
    detections_per_letter_avg = {}

    for _class in classes_of_interest:
        detections_per_letter[_class] = {}
        detections_per_letter_avg[_class] = {}

        for letter in alphabet:
            occurences = objects_per_class.map(
                lambda entry: count_class_for_letter(entry, letter, _class)
            ).sum()
            images_n = objects_per_class.map(
                lambda entry: count_images_for_letter(entry, letter)
            ).sum()
            detections_per_letter[_class][letter] = occurences
            detections_per_letter_avg[_class][letter] = occurences / images_n

    write_results("alphabet_count", ["letter", "count"], detections_per_letter)
    write_results(
        "alphabet_count_avg", ["letter", "avg_count"], detections_per_letter_avg
    )

if "--skip2" not in sys.argv:
    cities = ["New York", "Los Angeles", "Detroit", "Paris", "Berlin", "Warsaw"]

    # return number of detected objects of class _class, but count them only if landmark's name starts with letter
    def count_class_for_city(entry, city, _class):
        id = entry["landmark_id"]
        name = landmark_names.at[id, "name"]

        if city not in name:
            return 0

        detections = json.loads(entry["predictions_sum"])
        return detections.get(str(_class), 0)

    def count_files_for_city(entry, city):
        id = entry["landmark_id"]
        name = landmark_names.at[id, "name"]

        if city not in name:
            return 0

        return int(entry["image_count"])

    detections_per_city_avg = {}

    for _class in classes_of_interest:
        detections_per_city_avg[_class] = {}
        for city in cities:
            detections = objects_per_class.map(
                lambda entry: count_class_for_city(entry, city, _class)
            ).sum()
            file_count = objects_per_class.map(
                lambda entry: count_files_for_city(entry, city)
            ).sum()
            detections_per_city_avg[_class][city] = detections / file_count

    write_results(
        "avg_obj_per_city", ["city", "avg_detections"], detections_per_city_avg
    )

if "--skip3" not in sys.argv:

    def count_people(entry):
        detections = json.loads(entry["predictions_sum"])
        return detections.get(str(0), 0)

    def count_images(entry):
        return int(entry["image_count"])

    total_people = objects_per_class.map(lambda entry: count_people(entry)).sum()
    total_files = objects_per_class.map(lambda entry: count_images(entry)).sum()

    print(total_files)
    print(total_people)

    people_classes = objects_per_class.filter(
        lambda entry: "people"
        in landmark_names.at[entry["landmark_id"], "name"].lower()
    )
    total_people_in_people = people_classes.map(lambda entry: count_people(entry)).sum()
    total_people_files = people_classes.map(lambda entry: count_images(entry)).sum()

    print(total_people_files)
    print(total_people_in_people)

    people_avg = total_people / total_files
    people_avg_people = total_people_in_people / total_people_files

    people = {0: {"avg_all": people_avg, "avg_people_places": people_avg_people}}

    write_results(
        "people_in_places_with_people", ["files considered", "avg_detections"], people
    )
