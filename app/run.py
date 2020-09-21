import json
import plotly
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
from sklearn.model_selection import train_test_split


app = Flask(__name__)


class SentenceLengthExtractor(BaseEstimator, TransformerMixin):

    def sen_len(self, text):
        """
        Calculates the length of the number of words in the message to use as a feature
        :param text: message
        :return: length of message
        """
        tokens = nltk.word_tokenize(text)
        return len(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.sen_len)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """
       This function takes in the message as the input and performs various steps like removing punctions,
       tokenizing into words, removing stop words as well as stemming and lemmatizing the words which can be used
       for the classifier
       :param text: message (input)
       :return: returns cleaned tokens
       """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterR.db')
df = pd.read_sql_table('test_table_5', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    X = df['message']
    y = df.iloc[:, 4:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_y = Y_train
    new_y = new_y.sum()
    col_names = new_y.index
    value_counts = new_y.values
    new_X = Counter(" ".join(X_train).split()).most_common(15)
    word_list = []
    freq_list = []
    for word, count in new_X:
        word_list.append(word)
        freq_list.append(count)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=col_names,
                    y=value_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Category Types',
                'yaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=word_list,
                    y=freq_list
                )
            ],

            'layout': {
                'title': 'Top 15 Words in Dataset',
                'yaxis': {
                    'title': "Count"
                },

            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()