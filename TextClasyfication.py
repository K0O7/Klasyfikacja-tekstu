import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure


def extract_categories(categories_lists):
    categoriesCount = {}
    categories = categories_lists
    for cat in categories:
        if cat is not None and type(cat) == str and len(cat) > 0:
            values = eval(cat).values()
            for val in values:
                if val in categoriesCount:
                    categoriesCount[val] = categoriesCount[val] + 1
                else:
                    categoriesCount[val] = 1
    return pd.DataFrame(data=categoriesCount.values(), index=categoriesCount.keys(), columns=['values'])


def category_analyze(data):
    print("counting missing data")
    print(data.isna().sum())
    categories = extract_categories(data["categories"])
    categories = categories.sort_values(by="values", ascending=False )
    figure(figsize=(40, 15), dpi=100)
    sns.barplot(x=categories.index, y="values", data=categories)
    plt.xticks(rotation=90)
    plt.show()


def data_selection(data, selected_categories):
    data = data[data['categories'].notnull()]
    data = data[data['categories'].str.contains("|".join(selected_categories))]
    data, repeated_categories = normalized_data(data, selected_categories)
    print("powtórzenia w kategoriach")
    print(repeated_categories)
    plot_data = data.groupby(['category']).count()
    p_d = plot_data.sort_index()
    sns.barplot(x=p_d.index, y="description", data=p_d)
    plt.title("Number of books in category")
    plt.show()
    print(plot_data)
    rc = repeated_categories.sort_index()
    sns.barplot(x=rc.index, y="value", data=rc)
    plt.title("Number of categories with another categories")
    plt.show()
    return data.rename(columns={'category': 'label', 'description': 'text'})


def normalized_data(data, categories):
    result_data = []
    repeated_categories = {}
    data.reset_index()
    for index, row in data.iterrows():
        rep = []
        for category in categories:
            if category in row['categories']:
                rep.append(category)
        if len(rep) > 1:
            for r in rep:
                if r in repeated_categories:
                    repeated_categories[r] = repeated_categories[r] + 1
                else:
                    repeated_categories[r] = 1
        else:
            result_data.append([rep[0], row['description']])
    return pd.DataFrame(data=result_data, columns=['category', 'description']), pd.DataFrame(
        data=repeated_categories.values(), index=repeated_categories.keys(), columns=['value'])
