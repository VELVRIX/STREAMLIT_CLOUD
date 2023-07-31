import streamlit as st
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import tensorflow as tf
from PIL import Image

# Suppressing scientific notation
np.set_printoptions(suppress=True)



#@st.cache(allow_output_mutation=True) # st.cache is now deprecated 
@st.cache_resource
def get_model():
    model_x = AutoModelForSequenceClassification.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    tokenizer_x = AutoTokenizer.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    return tokenizer_x,model_x


tokenizer,model = get_model()

st.title('Sentiment analyzer')
st.caption('This classifier is Negativly Biased.')

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

st.divider()

'''
* This model is fine tunned version of RoBERTa-base & has a MCC score of 0.86.

* model can be accessed from Hugging Face : https://huggingface.co/velvrix/truefoundary_sentimental_RoBERTa
'''

st.divider()

'''
* ``Try new Multi-Class Classification model`` -> 3 different class with Positive, Neutral and Negative sentiments : https://huggingface.co/velvrix/RoBERTA_pipeiq
'''



image = Image.open('Screenshot 2023-07-31 183620.jpg')

st.image(image, caption='Multi-Class Classification Unbiased model')
