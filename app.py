from tensorflow import keras
import os
import openai
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st



st.markdown("<p>hi</p>", unsafe_allow_html=True)

openai.api_key = st.secrets.OPENAI_TOKEN

df=pd.read_csv("./data/mackerelsalted.csv", encoding="utf8")
df = df.rename(columns={df.columns[1]: 'mackerel'})
df['mackerel'] = df['mackerel'].str.replace(',', '').astype(float)

dataset = df['mackerel'].values.reshape(-1, 1)
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

look_back = 3


def predict_mackerel_price_for_tomorrow():
    # 최근 3일의 데이터를 가져와서 예측 진행
    last_data = dataset[-look_back:]
    last_data_reshaped = np.reshape(last_data, (1, 1, look_back))
    
    # 모델로 내일 값을 예측
    predicted_value = model.predict(last_data_reshaped)
    
    # 예측값을 원래의 스케일로 변환
    predicted_value = scaler.inverse_transform(predicted_value)
    
    return predicted_value[0][0]

def request_chat_completion(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response["choices"][0]["message"]

model = keras.models.load_model('md/my_model.keras')

# 예측값 가져오기
prediction = predict_mackerel_price_for_tomorrow()

# 스트림릿에 예측값 출력하기
st.markdown(f"<p>Predicted Price for Tomorrow: {prediction:.2f}</p>", unsafe_allow_html=True)


with st.form("form"):
    question = st.text_input("질문", "내일 고등어 가격은 얼마인가요?")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        if "고등어 가격" in question:
            predicted_price = predict_mackerel_price_for_tomorrow()
            st.markdown(f"<p>Predicted Price for Tomorrow: {predicted_price:.2f}</p>", unsafe_allow_html=True)
            
            messages = [{"role": "user", "content": f"내일 고등어 가격은 {predicted_price:.2f}로 도출됩니다. {question}"}]
            mackerel_response = request_chat_completion(messages)
            st.write(f"답변: {mackerel_response['content']}")
#     question = request.form['question']
#     predicted_price = predict_mackerel_price_for_tomorrow()
    
#     messages = [{"role": "user", "content": f"내일 고등어 가격은 {predicted_price:.2f}로 도출됩니다. {question}"}]
#     response = request_chat_completion(messages)
    
#     return render_template('index.html', prediction=response["content"])





