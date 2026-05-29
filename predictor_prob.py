# predictor.py (versi SavedModel)
import pickle
import os
import numpy as np
from deep_translator import GoogleTranslator
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing_noprob import clean_text

MAX_LENGTH = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "results", "artifacts", "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "results", "artifacts", "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# SavedModel load dari foldernya, bukan file .pb langsung
model = tf.saved_model.load(os.path.join(BASE_DIR, "results","best_model"))
infer = model.signatures["serving_default"]


def predict(text: str) -> dict:

    text_in_to_eng = GoogleTranslator(source="auto", target="en").translate(text)
    text_clean = clean_text(text_in_to_eng)

    seq    = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    tensor = tf.constant(padded, dtype=tf.int32)


    output = infer(tensor)

    output_keys = list(output.keys())
    print("Output keys:", output_keys)  

    
    pred_emotion  = output[output_keys[0]].numpy()
    pred_healing  = output[output_keys[1]].numpy()

    probs = pred_emotion[0]
    idx   = int(np.argmax(probs))

    return {
        "text_original" : text,
        "text_english"  : text_in_to_eng,
        "text_clean"    : text_clean,
        "emotion"       : le.classes_[idx],
        "confidence"    : round(float(probs[idx]), 4),
        "all_emotions"  : {
            le.classes_[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
        "healing_score" : round(float(pred_healing[0][0]), 4),
    }