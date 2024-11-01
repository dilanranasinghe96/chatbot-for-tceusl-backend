import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import json
import random  # Add this import

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('models/chatbot_model.h5')

# Load the training data
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Unpack the dictionary
words = data['words']
classes = data['classes']

# Load the intents file
with open('data/augmented_intents.json') as f:
    intents = json.load(f)

def clean_up_sentence(sentence):
    """
    Tokenize and lemmatize the input sentence.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """
    Create a bag of words array.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    """
    Predict the class of the input sentence.
    """
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intent):
    """
    Get a response for the given intent from the intents file.
    """
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "Sorry, I didn't get that."

def chatbot_response(msg):
    """
    Generate a response from the chatbot based on the input message.
    """
    intents_list = predict_class(msg, model)
    if intents_list:
        intent = intents_list[0]['intent']
        return get_response(intent)
    return "Sorry, I didn't understand that."