from pathlib import Path
import plotly.express as px
import pandas as pd
from string import ascii_uppercase
from coco_classes import COCO_CLASSES

# classes_of_interest = [0, 1, 2, 11, 13]
classes_of_interest = [*range(0, len(COCO_CLASSES))]


def plots(type):
    Path(f"./plots/{type}").mkdir(parents=True, exist_ok=True)
    for _class in classes_of_interest:
        df = pd.read_csv(f"stats/{type}/{_class}.csv", header=0, sep=";")

        fig = px.bar(df, x="letter", y="count")
        fig.write_html(f"plots/{type}/{COCO_CLASSES[_class]}.html")


def heatmap(type):
    # Heatmap for all classes
    heatmap_df = pd.DataFrame({"letter": [*ascii_uppercase]})

    for _class in classes_of_interest:
        df = pd.read_csv(f"stats/{type}/{_class}.csv", header=0, sep=";")
        df = df.rename(columns={"count": COCO_CLASSES[_class]})
        heatmap_df = pd.merge(heatmap_df, df, on=["letter"])

    heatmap_df = heatmap_df.set_index(["letter"]).transpose()

    fig = px.imshow(heatmap_df, text_auto=True, aspect="auto")
    fig.write_html(f"plots/{type}/heatmap.html")


plots("alphabet_count")
plots("alphabet_count_avg")
heatmap("alphabet_count")
heatmap("alphabet_count_avg")
