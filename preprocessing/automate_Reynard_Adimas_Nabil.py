import pandas as pd 
import numpy as np 
import re 
import string
import nltk 
import os
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from joblib import dump 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

 

class TextPreprocessor(BaseEstimator, TransformerMixin): 

    def __init__(self, slang_dict=None):
        self.slang_dict = slang_dict if slang_dict is not None else {} 
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.list_stopwords = set(stopwords.words('english'))
        self.list_stopwords.update(["n't", "'s", "'m", "'re", "...", "u", "ur"]) 

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        return X.apply(self._process_text) 
    
    def _process_text(self, text):
        text = str(text)
        text = text.encode('ascii','ignore').decode('ascii')
        text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
        text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
        text = re.sub(r'RT[\s]', '', text) # remove RT
        text = re.sub(r"http\S+", '', text) # remove link
        text = re.sub(r'[0-9]+', '', text) # remove numbers
        text = text.replace('\n', ' ') # replace new line into space
        text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
        text = text.strip(' ') # remove characters space from both left and right text 

        # Case Folding 
        text = text.lower() 

        # Replace slang 
        if self.slang_dict:
            words = text.split()
            normalized_words = [self.slang_dict.get(word, word) for word in words]
            text = " ".join(normalized_words) 
        
        # Tokenizing & Stopwords 
        tokens = word_tokenize(text)
        filtered_tokens = [t for t in tokens if t not in self.list_stopwords] 

        # lemmatizing 
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]

        return ' '.join(lemmatized_words)


def preprocess_data(data, target_column, text_column, slang_dict_path, save_path):
    print("Mulai preprocessing")
    
    try:
        slang_df = pd.read_csv(slang_dict_path, header=0, names=['slang', 'formal'])
        slang_dict = dict(zip(slang_df['slang'], slang_df['formal']))
        print(f"Slang dictionary load: {len(slang_dict)}")
    except Exception as e:
        print(f"Gagal load slang dict")
        slang_dict = {} 
    
    print("Labelling")
    def labelling(rating):
        if(rating >= 4):
            return 1 
        else:
            return 0 
    
    data = data.dropna(subset=[text_column, target_column]).copy() 

    y = data[target_column].apply(labelling)
    X = data[text_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    ) 
    print(f"Train Size: {len(X_train)}")
    print(f"Test Size: {len(X_test)}") 

    nlp = Pipeline(steps=[
        ('preprocessor', TextPreprocessor(slang_dict=slang_dict)), 
        ('tfidf', TfidfVectorizer(max_features=200, min_df=17, max_df=0.8))
    ]) 

    X_train = nlp.fit_transform(X_train)
    X_test = nlp.transform(X_test) 
    dump(nlp, save_path)
    feature_names = nlp.named_steps['tfidf'].get_feature_names_out()
    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)

    return X_train_df, X_test_df, y_train, y_test 

if __name__ == "__main__":
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path_raw_data = os.path.join(base_path, 'whatsapp_review_raw', 'whatsapp_reviews.csv')
    slang_path = os.path.join(base_path, 'whatsapp_review_raw', 'acrynom.csv')

    output_dir = os.path.join(base_path, 'preprocessing', 'whatsapp_review_preprocessing')
    pipeline_save_path = os.path.join(output_dir, 'preprocess_pipeline.joblib')

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(path_raw_data):
        print(f"Error: File not found")
        sys.exit(1) 
    
    print(f"Loadin data from {path_raw_data}")
    df = pd.read_csv(path_raw_data)

    X_train, X_test, y_train, y_test = preprocess_data(
        data=df, 
        target_column='rating', 
        text_column='review_text', 
        slang_dict_path=slang_path,
        save_path=pipeline_save_path
    ) 
    print("Saving processed datasets")
    train_set = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_set = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    train_set.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    test_set.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

    print(f"Done, data tersimpan di {output_dir}")

   


