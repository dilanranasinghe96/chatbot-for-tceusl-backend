from flask import Flask, request, jsonify
from flask_cors import CORS  
from chatbot import chatbot_response 
from googletrans import Translator

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to handle language translation and response generation
def get_response(user_input, selected_language):
    translator = Translator()
    
    # Translate user input to English if needed
    if selected_language != 'english':
        user_input = translator.translate(user_input, src=selected_language, dest='english').text
    
    # Get response using your NLTK model (in English)
    response_in_english = chatbot_response(user_input)  # Assuming chatbot_response is your NLTK model function
    
    # Translate the response back to the selected language if necessary
    if selected_language != 'english':
        response_in_selected_language = translator.translate(response_in_english, src='english', dest=selected_language).text
    else:
        response_in_selected_language = response_in_english
    
    return response_in_selected_language

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message')
    selected_language = data.get('language', 'english')  # Default to English if no language is provided
    
    # Get response in the appropriate language
    response = get_response(message, selected_language)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
