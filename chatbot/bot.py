import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data files (only need to run once)
nltk.download('stopwords')
nltk.download('punkt')


class Chatbot:
    def __init__(self):
        self.context = None

    def preprocess_input(self, user_input):
        tokens = nltk.word_tokenize(user_input.lower())
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(filtered_tokens)

    def get_response(self, user_input):
        patterns = [
            (r'hi|hello|hey', "Hello! How can I help you today?"),
            (r'how are you', "I'm just a bunch of code, but I'm doing great! How about you?"),
            (r'what is your name|who are you', "I am an advanced rule-based chatbot created to assist you."),
            (r'what can you do|what are your features',
             "I can respond to basic greetings and questions. I'm here to assist you with simple tasks."),
            (r'tell me a joke', "Why don't scientists trust atoms? Because they make up everything!"),
            (r'bye|exit|quit', "Goodbye! Have a great day!")
        ]

        for pattern, response in patterns:
            if re.search(pattern, user_input):
                if 'how are you' in pattern and self.context == 'asked_how':
                    response = "I already told you, I'm doing great! How about you?"
                self.context = pattern
                return response
        return "I'm sorry, I don't understand that. Can you please rephrase?"

    def run(self):
        print("Hello! I am an advanced chatbot. How can I assist you today?")

        while True:
            user_input = input("You: ").strip().lower()
            preprocessed_input = self.preprocess_input(user_input)

            response = self.get_response(preprocessed_input)

            if response == "Goodbye! Have a great day!":
                print("Chatbot: " + response)
                break
            else:
                print("Chatbot: " + response)


# Run the chatbot
chatbot = Chatbot()
chatbot.run()
