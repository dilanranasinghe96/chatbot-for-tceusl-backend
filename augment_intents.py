import json
import nltk
from nltk.corpus import wordnet
import random

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Function to get synonyms of a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)  # Use set to avoid duplicates

# Function to augment a pattern by replacing words with their synonyms
def augment_pattern(pattern):
    words = nltk.word_tokenize(pattern)
    augmented_patterns = []
    for word in words:
        synonyms = get_synonyms(word)
        for synonym in synonyms:
            new_pattern = pattern.replace(word, synonym)
            augmented_patterns.append(new_pattern)
    return augmented_patterns

# Load the intents JSON file
with open('data/intents.json', 'r') as file:
    intents = json.load(file)

# Augment the dataset
augmented_intents = {"intents": []}
for intent in intents['intents']:
    new_patterns = []
    for pattern in intent['patterns']:
        # Add the original pattern
        new_patterns.append(pattern)
        # Add augmented patterns
        new_patterns.extend(augment_pattern(pattern))
    
    # Remove duplicates and shuffle the patterns
    new_patterns = list(set(new_patterns))
    random.shuffle(new_patterns)

    # Update the intent with augmented patterns
    augmented_intent = {
        "tag": intent['tag'],
        "patterns": new_patterns,
        "responses": intent['responses']
    }
    augmented_intents['intents'].append(augmented_intent)

# Save the augmented intents back to a new JSON file
with open('data/augmented_intents.json', 'w') as outfile:
    json.dump(augmented_intents, outfile, indent=4)

print("Dataset augmentation complete. Augmented dataset saved as 'augmented_intents.json'")
