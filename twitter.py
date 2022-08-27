import requests
import re
import os
import json
import sys
import random

from PIL import Image
from min_dalle import MinDalle
import torch

from textblob import TextBlob
from summarizer import TransformerSummarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
MODEL = T5ForConditionalGeneration.from_pretrained('t5-base')
TOKENIZER = T5Tokenizer.from_pretrained('t5-base')

# TODO
# SUMMARIZER = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    print("saving image to", path)
    image.save(path)
    return image


def generate_image(
    is_mega: bool,
    text: str,
    seed: int,
    grid_size: int,
    top_k: int,
    image_path: str,
    models_root: str,
    fp16: bool,
):
    model = MinDalle(
        is_mega=is_mega, 
        models_root=models_root,
        is_reusable=False,
        is_verbose=True,
        dtype=torch.float16 if fp16 else torch.float32
    )

    image = model.generate_image(
        text,
        seed,
        grid_size,
        top_k=top_k,
        is_verbose=False
    )
    save_image(image, image_path)



def summarize_tweets(tweets):
    TEXT_CLEANING_RE = "#\S+|@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text_clean = re.sub(TEXT_CLEANING_RE, ' ', str(".".join(tweets)).lower()).strip()

    words = text_clean.split(" ")
    removed_duplicates = []
    for w in words:
        if w not in removed_duplicates:
            removed_duplicates.append(w)
    tokens_input = TOKENIZER.encode("summarize: " + " ".join(removed_duplicates), return_tensors="pt", max_length=512, truncation=True)
    summary_ids = MODEL.generate(
        tokens_input,
        min_length=10,
        max_length=100,
        length_penalty=4.0)
    summary = TOKENIZER.decode(summary_ids[0])
    return summary

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
TOKEN = open('./.cred', 'r+').read()

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

QUERY = 'kelowna'

def get_latest_tweets(url, query):
    query_params = {'query': f'#{query}','tweet.fields': 'author_id', 'max_results': '20'}
    headers = {'Authorization': f"Bearer {TOKEN}", 'User-Agent': 'Python4.8'}
    response = requests.get(url, headers=headers, params=query_params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else QUERY
    query_cache_name = f'{query}_cache.json'
    if not os.path.exists(query_cache_name):
        print('hitting twitter')
        json_response = get_latest_tweets(SEARCH_URL, query=query)
        json.dump(json_response, open(query_cache_name, 'w+'), indent=4, sort_keys=True)
    else:
        print('loading cache')
        json_response = json.load(open(query_cache_name, 'r'))
    
    # take out keywords here if you don't want them summarized
    # I guess I think there is enough information on these topics out there :)
    keywords = ['police', 'crime', 'drug']
    tweet_text = [d['text'] for d in json_response['data'] if not any([k in d['text'] for k in keywords])]
    summary = summarize_tweets(tweet_text)
    
    # TODO how can something more cohesive be generated (without more training data...?)
    printable_string = summary.lstrip('<pad> ').replace('< /s >', '')
    print(printable_string)
    blob = TextBlob(printable_string)
    nouns = blob.noun_phrases
    print(nouns)
    sampled = random.sample(nouns, 3)
    sentence = f'A {sampled[0]} on a {sampled[1]}'
    print(sentence)

    # mindal image gen
    generate_image(
        is_mega=True,
        text=sentence,
        seed=-1,
        grid_size=1,
        top_k=256,
        image_path=query,
        models_root='pretrained',
        fp16=False,
    )

if __name__ == "__main__":
    main()