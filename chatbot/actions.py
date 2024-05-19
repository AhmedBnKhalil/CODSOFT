from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests


class ActionWeather(Action):

    def name(self) -> Text:
        return "action_weather"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        location = "Cairo"  # You can extract location from user input
        api_key = "d52f3bcf31f6c16a965cb6a50af4f163"
        weather_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
        response = requests.get(weather_url).json()

        if response:
            temp_c = response['current']['temp_c']
            condition = response['current']['condition']['text']
            weather_info = f"The current temperature in {location} is {temp_c}°C and the weather is {condition}."
        else:
            weather_info = "I'm sorry, I couldn't fetch the weather details at the moment."

        dispatcher.utter_message(text=weather_info)
        return []
