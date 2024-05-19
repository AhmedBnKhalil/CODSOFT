from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = get_rasa_response(user_message)
    return jsonify({'response': response})

def get_rasa_response(message):
    rasa_url = 'http://localhost:5005/webhooks/rest/webhook'  # Rasa server URL
    payload = {
        'sender': 'user',
        'message': message
    }
    response = requests.post(rasa_url, json=payload)
    if response.ok:
        response_data = response.json()
        if response_data:
            return response_data[0].get('text', '')
    return "I'm sorry, I didn't understand that."

if __name__ == '__main__':
    app.run(debug=True)
