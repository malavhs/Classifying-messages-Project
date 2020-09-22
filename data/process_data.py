import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the data from the two csv files mentioned, before merging it to form a single dataframe
    :param messages_filepath: Path to the disaster_messages.csv file based as command link argument
    :param categories_filepath: Path to the disaster_categories.csv file based as command link argument
    :return: merged dataframe
    """

    # Read in the two csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #Merge the two dataframes
    df = messages.merge(categories, how="left", on='id')

    return df


def clean_data(df):
    """
    This function inputs the merged dataframe and applies cleaning steps like column type conversions, dropping, concatenating etc.
    to create a clean dataframe
    :param df: raw merged dataframe
    :return: clean dataframe with no duplicates, nulls etc.
    """


    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = []
    for col_name in row:
        category_colnames.append(col_name[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop old categories column
    df = df.drop('categories', axis=1)

    #Concat new categorical columns with dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop rows that are 2 in the related column
    df = df[df.related != 2]

    #Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    This function saves the cleaned dataframe in a sqlite database
    :param df: cleaned dataframe
    :param database_filename: name of the databased based as command line argument
    :return: None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('test_table_5', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterR.db')


if __name__ == '__main__':
    main()