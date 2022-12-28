import streamlit as st
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import tensorflow as tf


# Suppressing scientific notation
np.set_printoptions(suppress=True)

@st.cache(allow_output_mutation=True)
def get_model():
    model_x = AutoModelForSequenceClassification.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    tokenizer_x = AutoTokenizer.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    return tokenizer_x,model_x


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Postive Sentiment',
  0:'Negative Sentiment'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    predictions = tf.nn.softmax(output.logits.detach().numpy(), axis =1)
    show = predictions.numpy()
    st.write("Negative :", show[0][0], "Positive :", show[0][1])
    # st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])





'''
import streamlit as st
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import numpy as np

@st.cache(allow_output_mutation=True)
def get_model():
    model_x = AutoModelForSequenceClassification.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    tokenizer_x = AutoTokenizer.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    return tokenizer_x,model_x


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Postive Sentiment',
  0:'Negative Sentiment'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    # st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])

'''

