from pathlib import Path
import plotly.express as px
import pandas as pd
from string import ascii_uppercase
from coco_classes import COCO_CLASSES

# classes_of_interest = [0, 1, 2, 11, 13]
classes_of_interest = [*range(0, len(COCO_CLASSES))]
cities = ["New York", "Los Angeles", "Detroit", "Paris", "Berlin", "Warsaw"]


def plots(type, x="letter", y="count", classes_overwrite=[]):
    Path(f"./plots/{type}").mkdir(parents=True, exist_ok=True)
    if len(classes_overwrite) != 0:
        classes_to_iter = classes_overwrite
    else:
        classes_to_iter = classes_of_interest

    for _class in classes_to_iter:
        df = pd.read_csv(f"stats/{type}/{_class}.csv", header=0, sep=";")

        fig = px.bar(df, x=x, y=y)
        fig.write_html(f"plots/{type}/{COCO_CLASSES[_class]}.html")


def heatmap(type, index_column_name, index_values, column_name="count"):
    Path(f"./plots/{type}").mkdir(parents=True, exist_ok=True)
    # Heatmap for all classes
    heatmap_df = pd.DataFrame({index_column_name: index_values})

    for _class in classes_of_interest:
        df = pd.read_csv(f"stats/{type}/{_class}.csv", header=0, sep=";")
        df = df.rename(columns={column_name: COCO_CLASSES[_class]})
        heatmap_df = pd.merge(heatmap_df, df, on=[index_column_name])

    heatmap_df = heatmap_df.set_index([index_column_name]).transpose()

    fig = px.imshow(heatmap_df, text_auto=True, aspect="auto")
    fig.write_html(f"plots/{type}/heatmap.html")


plots("alphabet_count")
plots("alphabet_count_avg", "letter", "avg_count")
plots("people_in_places_with_people", "files considered", "avg_detections", [0])
heatmap("alphabet_count", "letter", [*ascii_uppercase])
heatmap("alphabet_count_avg", "letter", [*ascii_uppercase], "avg_count")
heatmap("avg_obj_per_city", "city", cities, "avg_detections")
