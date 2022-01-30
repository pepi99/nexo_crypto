__author__ = "Petar Ulev"

import nltk
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import re


# Check this: https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c
# Using NLTK's Vader pre-trained sentiment analyser allows us to run it on unlabelled data (as is ours)
# It uses some heuristics and words like "really, very" to increase the sentiment scores.
# Lexicon-based

def get_cryptocurrency_list():
    """

    :return: a dictionary containing many cryptocurrencies as well as their abbreviations
    """
    with open('data/cryptocurrencies.json') as f:
        data = json.load(f)
        CRYPTOCURRENCIES = list(data.keys())
        return CRYPTOCURRENCIES


def get_cryptos_to_analyse_dict():
    """

    :return: list of the full names of the coins to analyse (ex.: ['bitcoin', 'ethereum', 'cardano']
    """
    with open('data/cryptos_to_analyse.json') as f:
        CRYPTOCURRENCIES = json.load(f)
        return CRYPTOCURRENCIES


CRYPTOCURRENCIES = get_cryptos_to_analyse_dict()

CRYPTOCURRENCIES_SN = CRYPTOCURRENCIES.keys()  # Small names
CRYPTOCURRENCIES_FN = CRYPTOCURRENCIES.values()  # Full names

stopwords = nltk.corpus.stopwords.words("english")  # NLTK stop words
lemmatizer = WordNetLemmatizer()  # nltk wordnet lemmatizer


def get_cryptos_to_analyse():
    """

    :return: list of the full names of the coins to analyse (ex.: ['bitcoin', 'ethereum', 'cardano']
    """
    return list(CRYPTOCURRENCIES.values())


cryptos_to_analyse = get_cryptos_to_analyse()


def get_api_data():
    """
    :return: configuration file loaded into a dictionary
    """
    with open('data/api_data.json') as f:
        api_data = json.load(f)
        return api_data


def preprocess(text):
    """
    The preprocessing of the reddit data is simple - just remove the links. It is done this way, because the NLTK sentiment analyser is influenced
    by punctiation, emojis, capital letters and words with repeated letter. For that reason, minimal preprocessing is done on the texts.
    :param text: input text
    :return: preprocessed text
    """
    r = re.sub(r'http\S+', '', text)
    return r


def analyse_subreddit(subreddit, j):
    """
    This function makes an analysis of a subreddit (the title + all of the subreddit's top level comments).
    Techniques used: stopword removal, lemmatization using NLTK, frequency distribution analysis using NLTK, plotting.
    NOTE! Stemming (like PorterStemmer, for example) is not used, because it this case we want to inspect solely the words (or their root), not stemmed words.
    PorterStemmer is usually used when we vectorize the texts/words.
    :param subreddit: a subreddit object
    :return: some statistics
    """
    title = subreddit['title_data']['title']
    comments = subreddit['comments_data']['comments']

    full_text = ' '.join([title, *comments])
    full_text = full_text.lower()  # Lowercase the text

    tokens = word_tokenize(full_text)
    tokens = [w for w in tokens if w.isalpha()]  # Take only tokens that are made of letters
    tokens = [w for w in tokens if w.lower() not in stopwords]  # Ignore stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Lemmatize tokens

    frequencies = FreqDist(tokens)  # Word frequencies

    fig = plt.figure(figsize=(10, 8))
    frequencies.plot(30, cumulative=False, show=False,
                     title=f'Word frequencies of subreddit {j}')  # Don't show it, just save it to the file
    fig.savefig(f'visualization/word_frequencies/wf_{int(j)}.png')
    plt.close()
    crypto_count = {cryptos_to_analyse[0]: 0, cryptos_to_analyse[1]: 0, cryptos_to_analyse[2]: 0}

    for crypto in list(CRYPTOCURRENCIES.values()):
        if crypto in frequencies.keys():
            crypto_count[crypto] += frequencies[crypto]
    for crypto in list(CRYPTOCURRENCIES.keys()):
        crypto_fn = CRYPTOCURRENCIES[crypto]
        if crypto in frequencies.keys():
            # if crypto_fn in crypto_count.keys():
            crypto_count[crypto_fn] += frequencies[crypto]
            # else:
            #     crypto_count[crypto_fn] = frequencies[crypto]
    # print(crypto_count)
    return len(tokens), crypto_count, frequencies


def nltk_sentiment(text):
    """
    The compound score of the NLTK's Vader sentiment analyser is used. Check here the metrics: https://github.com/cjhutto/vaderSentiment#about-the-scoringv
    :param text: input
    :return: returns the sentiments using the NLTK's pre-trained sentiment analyser - Vader. The compound score is used in our case.
    """
    sia = SentimentIntensityAnalyzer()
    vs = sia.polarity_scores(text)
    compound = vs['compound']
    if compound >= 0.05:
        return 'positive'
    elif (compound > -0.05) and (compound < 0.05):
        return 'neutral'
    elif compound <= -0.05:
        return 'negative'


def check_crypto_types(subreddit, max_number_cryptos=3):
    """
    Method to check which cryptocurrencies are mostly mentioned in a specific subreddit
    :param subreddit: a dictionary with title and comment keys.
    :return: a dictonary with each cryptocurrency mentioned in the subreddit and its count.
    """
    print(subreddit)
    cryptocurrencies = {}
    full_text = ' '.join([subreddit['title'], *subreddit['comments']]).lower()
    tokens = word_tokenize(full_text)

    for crypto in CRYPTOCURRENCIES.values():
        if crypto in tokens:
            cryptocurrencies[crypto] = tokens.count(crypto)

    for crypto in CRYPTOCURRENCIES.keys():
        if crypto in tokens:
            if CRYPTOCURRENCIES[crypto] in cryptocurrencies.keys():
                cryptocurrencies[CRYPTOCURRENCIES[crypto]] += tokens.count(crypto)
            else:
                cryptocurrencies[CRYPTOCURRENCIES[crypto]] = tokens.count(crypto)
    if len(list(cryptocurrencies.keys())) > max_number_cryptos:
        return 'False'
    return cryptocurrencies


def get_crypto_from_text(text):
    """

    :param text: input text
    :return: which coins (from the ones that we are interested in) are included in the text.
    """
    text = text.lower()
    cryptocurrencies = {}
    tokens = word_tokenize(text)

    for token in tokens:  # go through all the tokens (which are words in this case)
        if token in CRYPTOCURRENCIES_FN:  # Check if in full names
            if token not in cryptocurrencies.keys():
                cryptocurrencies[token] = 1  # set to 1
            else:
                cryptocurrencies[token] += 1  # increase
        elif token in CRYPTOCURRENCIES_SN:  # In this case, we have to check if we have that cryptocurrency's full name already, we can do that with a look up on the dicitonary CRYPTOCURRENCIES
            token_fn = CRYPTOCURRENCIES[token]  # This is the crypto full name
            if token_fn in cryptocurrencies.keys():  # just add 1 to it
                cryptocurrencies[token_fn] += 1
            else:
                cryptocurrencies[token_fn] = 1

    return cryptocurrencies


def plot_pie_chart(df, crypto, pltt):
    """
    Creates a pie chart and saves it.
    :param df: pd df
    :param crypto: type of crypto
    :param pltt: plot context
    :return:
    """
    ax = df.plot(kind='pie', title=f'Pie chart showing sentiment proportions for {crypto} for today',
                 autopct='%1.1f%%')
    fig = ax.get_figure()
    fig.savefig(f'visualization/sentiment_plots/piechart_{crypto}.png')
    pltt.show()
    pltt.close()


def summ(l):
    """
    Sums elements in list. Created because Python's sum built-in throws an error when summing some objects.
    :param l: list
    :return: summed elements in list
    """
    res = l[0]
    for x in l[1:]:
        res += x
    return res


def add_sentiment_title(crypto_list, sentiment_doc, sr):
    """
    This method adds the final sentiments for each coin that we want to analyse, considering title of subreddit.
    :param crypto_list: cryptos that the title talks about
    :param sentiment_dic: the sentiment dictionary
    :return: the sentiment dictionary with updated sentiments
    """
    for crypto in crypto_list:
        if crypto in cryptos_to_analyse:  # Check if crypto is included in what we want to analyse (the list of 3 cryptos)
            sentiment = sr['title_data']['sentiment']
            if crypto in sentiment_doc.keys():
                sentiment_doc[crypto][sentiment] += 1
            else:  # Then just create the sentiment
                sentiment_doc[crypto] = {'positive': 0, 'negative': 0, 'neutral': 0}
                sentiment_doc[crypto][sentiment] += 1
    return sentiment_doc


def add_sentiment_comment(comments_cryptos, sentiment_doc, sr):
    """
    This method adds the final sentiments for each coin that we want to analyse, considering comments of subreddit.
    :param comments_cryptos: the list of cryptos for the comments for a subreddits
    :param sentiment_doc: the sentiment dictionary
    :return: the sentiment dictionary with updated sentiments
    """

    for j, comment_cryptos in enumerate(comments_cryptos):
        for comment_crypto in comment_cryptos:  # There can be multiple cryptos for the same comment
            if comment_crypto in cryptos_to_analyse:  # Then we know that we should analyse it
                sentiment = sr['comments_data']['sentiments'][
                    j]  # get the sentiment for that specific comment with these crypto types
                if comment_crypto in sentiment_doc.keys():
                    sentiment_doc[comment_crypto][sentiment] += 1
                else:
                    sentiment_doc[comment_crypto] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    sentiment_doc[comment_crypto][sentiment] += 1
    return sentiment_doc
