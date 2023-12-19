import os
import json
import requests
from transformers import AutoTokenizer
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()  # take environment variables from .env.


API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
API_TOKEN = os.environ["HF_API_TOKEN"]

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


def query_hf(prompt):
    payload = {"inputs": prompt}
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json",}
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    json_response = json.loads(response.content.decode("utf-8"))
    print(json_response)
    if 'error' in json_response:
       raise Exception(json_response['error'])
    elif json_response and len(json_response) > 0:
      return json_response[0]["generated_text"]
    else:
      print(json_response)
      return ''
    

def chat_infer(chat: List[Dict]):
    """
    Chat should be in a list of dicts, alternating user / assistant roles e.g.

    chat = [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital is New Delhi"},
    ]

    """
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    print(prompt)
    response = query_hf(prompt)
    return response

def chat_infer2(question):
    template = f'<s>[INST]{question}[/INST]'.format(question)
    response = query_hf(template)
    return response






