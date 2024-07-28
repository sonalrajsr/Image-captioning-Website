from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import io
import json


with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Function to map indices back to words
# def index_to_word(index, tokenizer):
#     for word, idx in tokenizer.word_index.items():
#         if idx == index:
#             print(word)
#             return word
#     return None

# istoken = index_to_word(909, tokenizer)


def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))  # Resize as required by your model
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    print("\n \n \n \n ------------------------------image is processed successfully ----------------------\n\n\n\n\n\n")
    return img

# Use raw string or double backslashes for file path
file_path = r'static/uploads/8546bbf3-ac58-4bf7-af55-d5874db11ae2.jpeg'

# Open the image file in binary mode
with open(file_path, 'rb') as file:
    image = file.read()

# Preprocess the image
processed_image = preprocess_image(image)
print(processed_image)