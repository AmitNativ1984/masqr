# 1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E           https://245fd77d9c53.ngrok.io/

# python bot.py
# ./ngrok http 8080
# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/getWebhookInfo
# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/setWebHook?url=https://ec2a3aa82483.ngrok.io/

# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/sendMessage?chat_id=530533958&text=123456

import json
import urllib

import requests
import time

TOKEN = "1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js


def echo_all(updates):
    for update in updates["result"]:
        try:
            text = update["message"]["text"]
            chat = update["message"]["chat"]["id"]
            send_message(text, chat)
        except Exception as e:
            print(e)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def send_message(text, chat_id):
    text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text=123&chat_id=530533958".format(text, chat_id)
    get_url(url)


def main():
    last_update_id = None
    while True:
        print("getting updates")
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates)
        time.sleep(0.5)

