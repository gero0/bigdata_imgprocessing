from pathlib import Path
import plotly.express as px
import pandas as pd

classes_of_interest = [0, 1, 2, 11, 13]

Path("./plots/alphabet_count").mkdir(parents=True, exist_ok=True)
for _class in classes_of_interest:
    df = pd.read_csv(f'stats/alphabet_count/{_class}.csv', header=0, sep=';')

    fig = px.bar(df, x='letter', y='count')
    fig.write_html(f"plots/alphabet_count/{_class}.html")


Path("./plots/alphabet_count_avg").mkdir(parents=True, exist_ok=True)
for _class in classes_of_interest:
    df = pd.read_csv(f'stats/alphabet_count_avg/{_class}.csv', header=0, sep=';')

    fig = px.bar(df, x='letter', y='count')
    fig.write_html(f"plots/alphabet_count_avg/{_class}.html")