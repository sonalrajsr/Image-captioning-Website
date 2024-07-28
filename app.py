from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Add

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads' 
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

# Opening all captions files
with open('all_captions.pkl', 'rb') as f:
    all_captions = pickle.load(f)
# Load tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_len_cap = max(len(caption.split()) for caption in all_captions)


# Define Model
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

# VGG16 model for feature extraction
vgg16 = VGG16(weights='imagenet')
vgg16 = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)

max_length = 35  # Max Length of Caption


# Processing the uploaded images
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Extract Feature using VGG16
def extract_features(image):
    features = vgg16.predict(image)
    return features

# Function to convert predicted index to word form tokenizer 
def index_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

# Predict the caption 
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
    return in_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = file.read()
    processed_image = preprocess_image(image)
    features = extract_features(processed_image)
    caption = predict_caption(model, features, tokenizer, max_length)
    #Removing startseq and endseq throung slicing
    caption = caption[8:-6]
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
