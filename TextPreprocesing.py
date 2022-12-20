import re, string

import seaborn as sns
from tqdm import tqdm
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('words')


def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


wl = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)


words = set(nltk.corpus.words.words())


def language_filter(string):
    return " ".join(w for w in nltk.word_tokenize(string) if w in words or not w.isalpha())


def finalpreprocess(string):
    # return lemmatizer(stopword(language_filter(preprocess(string))))
    return stopword(preprocess(string))


def finalpreporcess_helper(column):
    columns = []
    print("Standardize words")
    for i in tqdm(range(len(column))):
        columns.append(finalpreprocess(column[i]))
    return columns


def word_count(df, selected_categories):
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    fig, axi = plt.subplots(1, 4, figsize=(20, 8))
    for cat in range(len(selected_categories)):
        train_words = df[df['label'] == selected_categories[cat]]['word_count']
        axi[cat].hist(train_words, color='red')
        axi[cat].set_title(selected_categories[cat])
    fig.suptitle('Words per description')
    plt.show()

    for cat in selected_categories:
        df['char_count'] = df['text'].apply(lambda x: len(str(x)))
        print(cat, df[df['label'] == cat]['char_count'].mean())

    sns.barplot(x='label', y='char_count', data=df)
    plt.title("Mean number of words in category")
    plt.show()
    return df


def text_cleaning(df, labels):
    df['text'] = finalpreporcess_helper(df['text'])
    df = word_count(df, labels)
    df = df[df['word_count'] > 0]
    df.to_csv("cleanedData.csv", sep="\t")
