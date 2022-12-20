from sklearn.feature_extraction.text import CountVectorizer
from clasyfication import nb_01
from DataSource import load_data, load_normalized_data
from TextClasyfication import category_analyze, data_selection
from TextPreprocesing import text_cleaning


# ["Science Fiction", "Mystery", "Fantasy", "Historical novel"]["Fantasy", "Novel", "Children's literature", "Science Fiction"]

selected_categories = ["Science Fiction", "Mystery", "Fantasy", "Historical novel"]

# df = load_data()
# category_analyze(df)
# selected_data = data_selection(df, selected_categories)
# print(selected_data)
# text_cleaning(selected_data, selected_categories)
df = load_normalized_data()
# print(df['text'])
# vectorizer = CountVectorizer()
# vector = vectorizer.fit_transform(df['text'])
# print(vector)
# svm_01(df, selected_categories)
print(type(df["text"]))
nb_01(df, selected_categories, is_bayes=True, ngram_range=(2, 3))
# nb_01(df, selected_categories, is_bayes=True, k=10000)
# nb_01(df, selected_categories, is_bayes=False, k=1000)
# nb_01(df, selected_categories, is_bayes=False, k=10000)
# nb_01(df, selected_categories, is_bayes=True, ngram_range=(2, 3))
# nb_01(df, selected_categories, is_bayes=True, ngram_range=(1, 3))






