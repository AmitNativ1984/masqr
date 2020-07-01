# 1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E           https://245fd77d9c53.ngrok.io/

# python bot.py
# ./ngrok http 8080
# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/getWebhookInfo
# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/setWebHook?url=https://ec2a3aa82483.ngrok.io/

# api.telegram.org/bot1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E/sendMessage?chat_id=530533958&text=123456

import json
import urllib
import json
import requests
import time

TOKEN = "1240246316:AAGeM3EhQG1wffLeQWqvDule5UTIjkdJQ0E"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

class telegram(object):

    def __init__(self):
        self.counter = 0
        self.datbase = {}

    def update_database(self):
        with open('./data/data%d.txt' % self.counter) as file:
            data = json.load(file)
            self.database.update({data["text"]: data["chat"]["id"]})
            self.counter += 1


    def get_url(self, url):
        response = requests.get(url)
        content = response.content.decode("utf8")
        return content


    def get_json_from_url(self,url):
        content = self.get_url(url)
        js = json.loads(content)
        return js


    def get_updates(self,offset=None):
        url = URL + "getUpdates?timeout=100"
        if offset:
            url += "&offset={}".format(offset)
        js = self.get_json_from_url(url)
        return js


    def echo_all(self,updates):
        for update in updates["result"]:
            try:
                text = update["message"]["text"]
                chat = update["message"]["chat"]["id"]
                self.send_message(text, chat)
            except Exception as e:
                print(e)


    def get_last_chat_id_and_text(self,updates):
        num_updates = len(updates["result"])
        last_update = num_updates - 1
        text = updates["result"][last_update]["message"]["text"]
        chat_id = updates["result"][last_update]["message"]["chat"]["id"]
        return (text, chat_id)


    def get_last_update_id(self,updates):
        update_ids = []
        for update in updates["result"]:
            update_ids.append(int(update["update_id"]))
        return max(update_ids)


    def send_message(self,text, aruco_id):
        text = timestamp:status:aruco_id:aruco_id
        msg=text.split(":")
        timestamp=msg[0]
        status=msg[2]
        id1 = msg[3]
        id2 = msg[4]

        text = urllib.parse.quote_plus(text)
        url = URL + "sendMessage?text=123&chat_id=530533958".format(text, chat_id)
        self.get_url(url)


if __name__ == '__main__':
    last_update_id = None
    while True:
        print("getting updates")
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates)
        time.sleep(0.5)


