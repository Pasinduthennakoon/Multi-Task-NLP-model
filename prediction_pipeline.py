import pickle
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

with open('model/tokernizer.pkl', 'rb') as f:
    tokernizer = pickle.load(f)
    
model = load_model('model/model.h5')

max_length = 50
stop_words = set(stopwords.words('english'))

def preprocess_new_text(text):
    # 1. Tokenize and remove stopwords (matching your training logic)
    words = nltk.word_tokenize(text.lower())
    filtered = [word for word in words if word not in stop_words]
    clean_text = ' '.join(filtered)
    
    # 2. Convert text to sequences using the SAVED tokenizer
    sequence = tokernizer.texts_to_sequences([clean_text])
    
    # 3. Pad sequences to max_length=50
    padded = pad_sequences(sequence, maxlen=max_length, padding = 'post')
    
    return padded

def predict_content(text):
    # Process the text
    processed_input = preprocess_new_text(text)
    
    # Predict using the three-input dictionary format required by your model
    predictions = model.predict({
        'emotion_input' : processed_input,
        'violence_input' : processed_input,
        'hate_input' : processed_input
    })
    
    # Extract results (Model outputs a list: [emotion, violence, hate])
    emo_pred = np.argmax(predictions[0])
    vio_pred = np.argmax(predictions[1])
    hat_pred = np.argmax(predictions[2])
    
    emotion_labels_text = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    violence_labels_text = ['harmful_traditional_practices', 'physical_violence',  'economic_violence', 'emotional_violence', 'sexual_violence']
    hate_labels_text = ['offensive_speech', 'hate_speech', 'neither']
    
    return {
        "Emotion Index": emotion_labels_text[emo_pred],
        "Violence Category": violence_labels_text[vio_pred],
        "Hate Class": hate_labels_text[hat_pred]
    }