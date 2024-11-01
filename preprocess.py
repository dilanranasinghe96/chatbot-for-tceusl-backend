# import nltk
# from nltk.stem import WordNetLemmatizer
# import numpy as np
# import json
# import pickle

# nltk.download('punkt')
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()

# def load_intents(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def preprocess_data(intents):
#     words = []
#     classes = []
#     documents = []
#     ignore_words = ['?', '!']
    
#     for intent in intents['intents']:
#         for pattern in intent['patterns']:
#             word_list = nltk.word_tokenize(pattern)
#             words.extend(word_list)
#             documents.append((word_list, intent['tag']))
#             if intent['tag'] not in classes:
#                 classes.append(intent['tag'])
    
#     words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
#     words = sorted(list(set(words)))

#     classes = sorted(list(set(classes)))

#     return words, classes, documents

# def save_preprocessed_data(words, classes, file_name):
#     with open(file_name, 'wb') as file:
#         pickle.dump((words, classes), file)


import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import json
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Load stopwords
stop_words = set(stopwords.words('english'))

def load_intents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word in the pattern
            word_list = nltk.word_tokenize(pattern)
            
            # Remove stopwords and apply stemming/lemmatization
            filtered_words = [lemmatizer.lemmatize(w.lower()) for w in word_list if w.lower() not in stop_words and w not in ignore_words]
            # To use stemming instead, uncomment the line below:
            # filtered_words = [stemmer.stem(w.lower()) for w in word_list if w.lower() not in stop_words and w not in ignore_words]
            
            words.extend(filtered_words)
            documents.append((filtered_words, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    # Remove duplicates and sort the words and classes
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return words, classes, documents

def save_preprocessed_data(words, classes, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump((words, classes), file)

# Example usage
if __name__ == "__main__":
    intents = load_intents('data/augmented_intents.json')
    words, classes, documents = preprocess_data(intents)
    save_preprocessed_data(words, classes, 'data/preprocessed_data.pkl')
    print("Preprocessing complete. Saved words and classes.")
