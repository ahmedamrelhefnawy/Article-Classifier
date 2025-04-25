import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Define Cosntants
MAX_SEQUENCE_LENGTH = 1025
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OBJECTS_DIR = os.path.join(CURRENT_DIR, 'Objects')

# Set current directory as base directory
os.chdir(CURRENT_DIR)

# Load Pre-processors & Model
lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model(OBJECTS_DIR + '/lstm_classification_model.keras')
tokenizer = pickle.load(open(OBJECTS_DIR + '/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open(OBJECTS_DIR + '/label_encoder.pkl', 'rb'))

# Define Variables
word_dict = word_dict = {v:k for k, v in tokenizer.word_index.items()}


# Define functions
def process_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I) # Remove extra white space from text

    text = re.sub(r'\W', ' ', str(text)) # Remove all the special characters from text

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Remove all single characters from text

    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove any character that isn't alphabetical

    text = text.lower()

    words = word_tokenize(text)

    words = [lemmatizer.lemmatize(word) for word in words]

    stop_words = set(stopwords.words("english"))
    Words = [word for word in words if word not in stop_words]

    Words = [word for word in Words if len(word) > 3]

    indices = np.unique(Words, return_index=True)[1]
    cleaned_text = np.array(Words)[np.sort(indices)].tolist()

    return cleaned_text


def decode(seq):
  return ' '.join([word_dict[i] for i in seq])

class pipeline:
    def __init__(self, model= model, tokenizer= tokenizer, label_encoder= label_encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
    
    def predict(self, text: str) -> str:
        text = process_text(text)
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        
        pred = self.model.predict(padded)
        pred = np.argmax(pred, axis=1)
        
        label = self.label_encoder.inverse_transform(pred)
        return label[0]