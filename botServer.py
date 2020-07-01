import json

import requests
from bottle import Bottle, response, request as bottle_request


# 1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E           https://245fd77d9c53.ngrok.io/

# python botserver.py
# ./ngrok http 8080
# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/getWebhookInfo
# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/setWebHook?url=https://a9602a2ac9ac.ngrok.io/


class BotHandlerMixin:
    BOT_URL = None

    def get_chat_id(self, data):
        """
        Method to extract chat id from telegram request.
        """
        chat_id = data['message']['chat']['id']

        return chat_id

    def get_message(self, data):
        """
        Method to extract message id from telegram request.
        """
        message_text = data['message']['text']

        return message_text

    def send_message(self, prepared_data):
        """
        Prepared data should be json which includes at least `chat_id` and `text`
        """
        message_url = self.BOT_URL + 'sendMessage'
        requests.post(message_url, json=prepared_data)


class TelegramBot(BotHandlerMixin, Bottle):
    BOT_URL = 'https://api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/'

    def __init__(self, *args, **kwargs):
        super(TelegramBot, self).__init__()
        self.route('/', callback=self.post_handler, method="POST")
        self.counter = 0;

    def change_text_message(self, text):
        return text[::-1]

    def prepare_data_for_answer(self, data):
        message = self.get_message(data)
        answer = self.change_text_message(message)
        chat_id = self.get_chat_id(data)
        json_data = {
            "chat_id": chat_id,
            "text": answer,
        }

        return json_data

    def post_handler(self):
        data = bottle_request.json
        answer_data = self.prepare_data_for_answer(data)
        with open('data\data%d.txt' % self.counter, 'w') as outfile:
            json.dump(data, outfile)
        self.counter = self.counter + 1
        self.send_message(answer_data)

        return response


if __name__ == '__main__':
    app = TelegramBot()
    app.run(host='localhost', port=8080)
