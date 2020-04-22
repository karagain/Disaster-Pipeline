import sys

import re
import nltk 
nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from sqlalchemy import create_engine




def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    
    return df, messages, categories


def clean_data(df, messages, categories):
    categories = categories.categories.str.split(";", expand=True)
    row = categories.iloc[0,:]
    category_colnames = [name[:-2] for name in row]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.loc[df.related == 2, "related"] = 1

    df = df.drop_duplicates()
    return df



def save_data(df, database_filename):
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("disasterresponses", engine, index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()