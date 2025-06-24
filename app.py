from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load XGBoost sales prediction model
model_sales = joblib.load("xgb_model.pkl")
expected_columns = ['Store', 'CPI', 'Unemployment', 'Week', 'Temperature',
                    'Fuel_Price', 'Month', 'Year', 'Holiday_Flag']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        input_df = input_df[expected_columns]
        prediction = model_sales.predict(input_df, validate_features=False)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

# Chatbot temporarily disabled to reduce memory usage
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model_chat = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
# chat_history = []
# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.get_json(force=True)['message']
#         chat_history.append(user_input)
#         full_chat = " ".join(chat_history) + tokenizer.eos_token
#         input_ids = tokenizer.encode(full_chat, return_tensors='pt')
#         output_ids = model_chat.generate(
#             input_ids,
#             max_length=1000,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95
#         )
#         reply = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
#         chat_history.append(reply)
#         return jsonify({'response': reply})
#     except Exception as e:
#         return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)




