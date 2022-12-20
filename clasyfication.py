from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, SelectKBest


#eksperymenty:
#1 zmiana parametru alpha C i gamma
#2 zmiana parametru K
#3 zmiana rozmiarów test_size przy train_test_split i suffle (z i bez zeby zobaczyć błąd w sztuce)

#4 test wpływu TfidfTransformer na ostateczne wyniki
#5 testowanie ngramow
#6 bez lematyzacji i angielskiego alfabetu


def nb_01(df, labels, is_bayes=True, test_size=0.2, val_size=0.2, shuffle=True, seed=False, max_df=0.5, ngram_range=(1,1), use_idf=False, smooth_idf=False, sublinear_tf=False, k=5000, alpha=0.01, C=100, gamma=0.01):
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=test_size, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=shuffle)

    if is_bayes:
        pipe = Pipeline(steps=[
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('chi2', SelectKBest(score_func=chi2)),
            ('mnnb', MultinomialNB())])
        pgrid = {
            'vect__ngram_range': [ngram_range],
            'tfidf__use_idf': [use_idf],
            'tfidf__smooth_idf': [smooth_idf],
            'tfidf__sublinear_tf': [sublinear_tf],
            'chi2__k': [k],
            'mnnb__alpha': [alpha]
        }
    else:
        pipe = Pipeline(steps=[
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('chi2', SelectKBest(score_func=chi2)),
            ('SVM', SVC())])
        pgrid = {
            'chi2__k': [k],
            "SVM__C": [C],
            "SVM__gamma": [gamma]
        }

    gs = GridSearchCV(pipe, pgrid, cv=5, n_jobs=-1, verbose=0)
    classifier = gs.fit(X_train, y_train)
    print("Train", gs.score(X_train, y_train))
    print("Test", gs.score(X_test, y_test))
    print("Validation", gs.score(X_val, y_val))
    print(gs.best_params_)
    preds = gs.predict(X_test)

    print(classification_report(y_test, preds, target_names=labels))

    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            display_labels=labels,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.show()


# def svm_01(df, labels):
#     X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2,
#                                                         shuffle=True)
#     pipe_svm = Pipeline(steps=[
#         ('vect', CountVectorizer()),
#         ('tfidf', TfidfTransformer()),
#         ('chi2', SelectKBest(score_func=chi2)),
#         ('SVM', SVC())])
#     pgrid_svm = {
#         'chi2__k': [5500, 6000],
#         "SVM__C": [0.001, 0.1, 10, 100, 10e5],
#         "SVM__gamma": [0.1, 0.01]
#     }
#
#     gs_svm = GridSearchCV(pipe_svm, pgrid_svm, cv=5, n_jobs=-1, verbose=10)
#     classifier = gs_svm.fit(X_train, y_train)
#     print("Train", gs_svm.score(X_train, y_train))
#     print("Test", gs_svm.score(X_test, y_test))
#     print(gs_svm.best_params_)
#     preds_svm = gs_svm.predict(X_test)
#
#     print(classification_report(y_test, preds_svm, target_names=labels))
#
#     titles_options = [
#         ("Confusion matrix, without normalization", None),
#         ("Normalized confusion matrix", "true"),
#     ]
#     for title, normalize in titles_options:
#         disp = ConfusionMatrixDisplay.from_estimator(
#             classifier,
#             X_test,
#             y_test,
#             display_labels=labels,
#             cmap=plt.cm.Blues,
#             normalize=normalize,
#         )
#         disp.ax_.set_title(title)
#
#         print(title)
#         print(disp.confusion_matrix)
#     plt.show()
