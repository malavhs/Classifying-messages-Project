# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

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


def load_data(database_filepath):
    """
    This function loads the cleaned dataframe stored in the sql database and
    separates it into input (X), output(y) and the list of the category names (labels)
    :param database_filepath: location of sql database which is passed through the command line
    :return: X,y, category names : input, outputs, list of category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('test_table_5', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    This function takes in the message as the input and performs various steps like removing punctions,
    tokenizing into words, removing stop words as well as stemming and lemmatizing the words which can be used
    for the classifier
    :param text: message (input)
    :return: returns cleaned tokens
    """

    #remove punctuations
    tokens = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(tokens)
    # Initialize Lemmatizer

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Lemmatizing to reduce to the actual root
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Defines the pipeline of the model that is to be trained which is tuned by using GridSearchCV
    while testing in the notebook.

    The custom transformer reduced performance to a great extent and
    hence not including in the final pipeline that I have kept
    :return: the pipeline of the model which is to be trained
    """
    pipeline = Pipeline([
                ('c_vect', CountVectorizer(tokenizer=tokenize)),
                ('c_tfidf_trans', TfidfTransformer()),
                ('c_clf_adabost', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1))))

        ])


    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model and prints out the classification report i.e. values like precision, recall, accuracy etc.
    for each of the category i.e. each output
    :param model: The trained model
    :param X_test: The test dataset (input)
    :param Y_test: The test labels (outputs)
    :param category_names: The list of category names
    :return: None
    """
    y_pred = model.predict(X_test)
    Y_test_vals = Y_test.values.transpose()
    Y_pred_vals = y_pred.transpose()
    for l, p, c in zip(Y_test_vals, Y_pred_vals,category_names):
        print("Category Name: ",c)
        print(classification_report(l, p))
    accuracy = ((Y_test == y_pred).mean()).values.mean()
    print("Overall Accuracy: ",accuracy)


def save_model(model, model_filepath):
    """
    This function saves the model as a pickle file to the specified location
    :param model: the trained model
    :param model_filepath: the path to store the model passed using the command line
    :return: None
    """
    joblib.dump(model, model_filepath)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()