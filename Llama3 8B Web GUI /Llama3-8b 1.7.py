import streamlit as st
import subprocess
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import requests

# Настройка логирования для сохранения в файл
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Инициализация модели и токенизатора
model_name = "Orenguteng/Lexi-Llama-3-8B-Uncensored"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_model_and_tokenizer()
if device.type == 'cuda':
    model.half()

def get_weather_data(city="London"):
    api_url = f"http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q={city}"
    try:
        response = requests.get(api_url)
        data = response.json()
        weather = data['current']['condition']['text']
        return weather
    except requests.exceptions.RequestException as e:
        logging.error(f"Network or API error: {e}")
        st.error("Ошибка сети или API.")
        return None

def generate_text(prompt, max_length, temperature, top_k):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length, temperature=temperature, top_k=top_k, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def handle_send():
    user_input = st.session_state.input
    max_length = st.session_state.max_length
    temperature = st.session_state.temperature
    top_k = st.session_state.top_k

    if user_input:
        logging.info(f"User input: {user_input}")
        try:
            if "погода" in user_input.lower():
                city = user_input.split()[-1]
                weather = get_weather_data(city)
                response = f"Погода в городе {city}: {weather}" if weather else "Не удалось получить погоду."
            else:
                response = generate_text(user_input, max_length, temperature, top_k)
            st.session_state['history'].append(f"Вы: {user_input}")
            st.session_state['history'].append(f"Модель: {response}")
        except Exception as e:
            logging.error(f"General error: {e}")
            st.error(f"Ошибка при обработке запроса: {str(e)}")
        st.session_state.input = ""

def kill_process():
    """Завершение процесса и всех дочерних процессов."""
    try:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(os.getppid())])
    except Exception as e:
        st.error(f"Ошибка при попытке завершить процесс: {e}")

st.title("Llama3 8B Web GUI")

if 'history' not in st.session_state:
    st.session_state['history'] = []

user_input = st.text_input("Введите ваш запрос:", key="input")
st.button("Отправить", on_click=handle_send)

for message in st.session_state['history']:
    st.text_area("", value=message, height=100, disabled=True, key=message)

if st.button("Очистить историю"):
    st.session_state['history'] = []
    st.experimental_rerun()

if st.button("Завершить все процессы", on_click=kill_process):
    st.stop()

st.sidebar.title("Настройки")
st.sidebar.slider("Max Length", 50, 300, 200, key="max_length")
st.sidebar.slider("Temperature", 0.5, 1.0, 0.7, key="temperature")
st.sidebar.slider("Top K", 1, 100, 40, key="top_k")
