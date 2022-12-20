import pandas as pd


def load_data():
    df = pd.read_csv("booksummaries.txt", delimiter="\t",
                     names=['wiki_id', 'id', 'title', 'author', 'date', 'categories', 'description'])
    fdf = df.filter(items=['categories', 'description'])
    return fdf

def load_normalized_data():
    return pd.read_csv("cleanedData.csv", delimiter="\t")
