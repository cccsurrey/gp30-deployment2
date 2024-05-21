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


def load_response_times():
    try:
        df = pd.read_csv('model_interactions.csv', usecols=["Timestamp", "Response Time"])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception as e:
        st.error(f"Failed to read response times: {e}")
        return pd.DataFrame()

def plot_response_times(df):
    if not df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['Response Time'], marker='o', linestyle='-')
        plt.title('Response Times Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Response Time (seconds)')
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("No response time data to display.")

#Function to setup the logs ina csv file
def setup_csv_logger():
    with open('model_interactions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        #The headers will be written if not present
        if file.tell() == 0:
            writer.writerow(["Timestamp", "User Input", "Model Prediction", "Response Time"])

def log_to_csv(sentence, results, response_time):
    with open('model_interactions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), sentence, results, response_time])

setup_csv_logger()

st.title('Group 30 - DistilBERT')
st.write('This application uses DistilBERT to classify Abbreviations (AC) and Long Forms (LF)')

example_sentences = [
    "RAFs are plotted for a selection of neurons in the dorsal zone (DZ) of auditory cortex in Figure 1.",
    "Light dissolved inorganic carbon (DIC) resulting from the oxidation of hydrocarbons.",
    "Images were acquired using a GE 3.0T MRI scanner with an upgrade for echo-planar imaging (EPI)."
]

sentence = st.selectbox('Choose an example sentence or type your own below:', example_sentences + ['Custom Input...'])

if sentence == 'Custom Input...':
    sentence = st.text_input('Input your sentence here', '')

def merge_entities(sentence, entities):
    entities = sorted(entities, key=lambda x: x['start'])
    annotated_sentence = ""
    last_end = 0
    for entity in entities:
        annotated_sentence += sentence[last_end:entity['start']]
        annotated_sentence += f"<mark style='background-color: #ffcccb;'><b>{sentence[entity['start']:entity['end']]}</b> [{entity['entity_group']}]</mark>"
        last_end = entity['end']
    annotated_sentence += sentence[last_end:]
    return annotated_sentence

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

if st.button('Classify'):
    if sentence:
        API_URL = API_URL
        headers = headers
        response, response_time = send_request_with_retry(API_URL, headers, {"inputs": sentence})
        if response is not None:
            results = response.json()
            st.write('Results:')
            annotated_sentence = merge_entities(sentence, results)
            st.markdown(annotated_sentence, unsafe_allow_html=True)
            log_to_csv(sentence, results, response_time)
            
            df = load_response_times()
            plot_response_times(df)
        else:
            st.error("Unable to classify the sentence due to server issues.")
    else:
        st.error('Please enter a sentence.')

#Separate button to just plot the response time
if st.button('Show Response Times'):
    df = load_response_times()
    plot_response_times(df)
