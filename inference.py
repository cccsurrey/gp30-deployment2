import streamlit as st
import requests
import csv
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

##########################
SECRET = os.environ["api_secret"]
headers = {"Authorization": "Bearer " + SECRET}
API_URL = "https://api-inference.huggingface.co/models/cccmatthew/surrey-gp30"
##########################

def log_to_csv(sentence, results, response_time):
    with open('model_interactions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), sentence, results, response_time])

def send_request_with_retry(url, headers, json_data, retries=3, backoff_factor=1):
    """Send request with retries on timeouts and HTTP 503 errors."""
    for attempt in range(retries):
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            response_time = time.time() - start_time
            return response, response_time
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                st.info('Server is unavailable, retrying...')
            else:
                raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            st.info(f"Network issue ({str(e)}), retrying...")
        time.sleep(backoff_factor * (2 ** attempt))

    st.error("Failed to process request after several attempts.")
    return None, None

sentence = [
    "RAFs are plotted for a selection of neurons in the dorsal zone (DZ) of auditory cortex in Figure 1.",
    "Light dissolved inorganic carbon (DIC) resulting from the oxidation of hydrocarbons.",
    "Images were acquired using a GE 3.0T MRI scanner with an upgrade for echo-planar imaging (EPI)."
]

API_URL = API_URL
headers = headers
response, response_time = send_request_with_retry(API_URL, headers, {"inputs": sentence})

if response is not None:
    results = response.json()
    print(results)

else:
    st.error("Unable to classify the sentence due to server issues.")