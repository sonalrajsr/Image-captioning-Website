import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Add
from tensorflow.keras.preprocessing.text import Tokenizer

import io

# Load model and tokenizer
with open('all_captions.pkl', 'rb') as f:
    all_captions = pickle.load(f)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_len_cap = max(len(caption.split()) for caption in all_captions)

input1 = Input(shape=(4096,), name="image_input")
fe1 = Dropout(0.4, name="image_dropout")(input1)
fe2 = Dense(256, activation='relu', name="image_dense")(fe1)

input2 = Input(shape=(max_len_cap,), name="text_input")
se1 = Embedding(vocab_size, 256, mask_zero=True, name="text_embedding")(input2)
se2 = Dropout(0.4, name="text_dropout")(se1)
se3 = LSTM(256, return_sequences=False, name="text_lstm")(se2)

decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation='relu', name="decoder_dense")(decoder1)
outputs = Dense(vocab_size, activation='softmax', name="output_dense")(decoder2)

model = Model(inputs=[input1, input2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights('model/model_weights_epoch_1.h5')

vgg16 = VGG16(weights='imagenet')
vgg16 = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)

max_length = 35

def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_features(image):
    features = vgg16.predict(image)
    return features

def index_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text[8:-6]

def main():
    st.title("Image Captioning Project")
    
    st.write("Upload an image to predict a caption:")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        image = uploaded_file.read()
        processed_image = preprocess_image(image)
        features = extract_features(processed_image)
        caption = predict_caption(model, features, tokenizer, max_length)
        
        st.write(f"**Predicted Caption:** {caption}")

    st.write("""
        **About This Project**

        This application allows you to upload an image, and it predicts a caption for the image using a deep learning model trained on a dataset of images and their corresponding captions. It demonstrates the application of AI in understanding and describing visual content.

        Created as a project to showcase image captioning technology using Streamlit, TensorFlow, and Python.
    """)

if __name__ == "__main__":
    main()
