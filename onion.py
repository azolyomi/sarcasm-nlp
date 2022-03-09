import re
import json
from bson.json_util import dumps
import unicodedata
import pymongo 
import sys
import numpy as np
from pymongo import MongoClient
from bs4 import BeautifulSoup
from urllib.request import urlopen

latestUrl = "https://www.theonion.com/latest"
RELEVANCY_DAY_COUNT = 30

def isNewsInBrief(article):
    text = article.get_text()
    if (len(re.findall("News In Brief", text)) > 0): return True
    return False

def toJson(headline):
    return json.dumps({ "headline": headline, "sarcastic": 1, 'relevancy': RELEVANCY_DAY_COUNT })

def cleanMongoFind(jsonObj):
    wanted_keys = ["headline", "sarcastic", "relevancy"]
    return { key: jsonObj[key] for key in wanted_keys }

def dumpToDb(MongoConnectionString, JsonHeadlines):
    client = MongoClient(MongoConnectionString)
    db = client.data
    # Simple relevancy metric for database using a month-long period of 'relevancy' per headline
    # First, we go through and subtract 1 from relevancy in each table entry, since a day has passed since last run of the cron:
    db.headlines.update_many({}, {"$inc": {'relevancy': -1}})
    # Then, we drop all elements from the collection where relevancy is <= 0:
    db.headlines.delete_many({'relevancy': {'$lte': 0}})
    # Finally, we insert all NEW entries into the table with a default relevancy score of RELEVANCY_DAY_COUNT:
    for hl_string in JsonHeadlines:
        hl = json.loads(hl_string)
        db.headlines.replace_one({ 'headline': hl.get('headline'), 'sarcastic': hl.get('sarcastic') }, hl, upsert=True)

def fetchFromDb(MongoConnectionString):
    client = MongoClient(MongoConnectionString)
    db = client.data
    cursor_jsons = dumps(db.headlines.find())
    fetched_json = json.loads(cursor_jsons)
    cleaned_json = list(map(cleanMongoFind, fetched_json))
    return cleaned_json
    


def parseLatest(MongoConnectionString):
    def clean(html):
        subbed = re.sub("<.*?>", "", html)
        return unicodedata.normalize('NFKD', subbed).encode('ascii', 'ignore').decode()
    latestPage = urlopen(latestUrl)
    latestHtml = latestPage.read().decode('utf-8')
    soup = BeautifulSoup(latestHtml, 'html.parser')
    headlinesJson = []
    for article in soup.find_all('article'):
        if (isNewsInBrief(article)): 
            headlinestemp = article.find('h2')
            cleaned = map(clean, headlinestemp)
            parsed = map(toJson, cleaned)
            for e in parsed:
                headlinesJson.append(e)

    # for e in headlineJson:
    #     print(e)
    dumpToDb(MongoConnectionString, headlinesJson)
