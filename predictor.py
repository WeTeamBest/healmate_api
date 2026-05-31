# predictor.py (versi SavedModel)
import pickle
import os
import numpy as np
from deep_translator import GoogleTranslator
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing_noprob import clean_text
import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# print(f"[DEBUG] API KEY: {os.getenv('GEMINI_API_KEY', 'TIDAK TERBACA')}")

MAX_LENGTH = 200

# ── Konfigurasi Gemini ──────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6LWjb9nVbt1pYqx6KNTLymv5Fu907YVzCIhyErCnpMuH")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "results", "artifacts", "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "results", "artifacts", "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# SavedModel load dari foldernya, bukan file .pb langsung
model = tf.saved_model.load(os.path.join(BASE_DIR, "results", "best_model"))
infer = model.signatures["serving_default"]


# ── Emotion label context (untuk prompt yang relevan) ───────
EMOTION_CONTEXT = {
    "anxiety": (
        "sedang merasa cemas dan khawatir berlebihan, "
        "gelisah memikirkan hal-hal yang tidak pasti setelah putus cinta"
    ),
    "anger": (
        "merasa marah, kecewa, dan sakit hati — mungkin merasa dikhianati "
        "atau diperlakukan tidak adil oleh mantan"
    ),
    "acceptance": (
        "mulai bisa menerima kenyataan putus cinta, "
        "meski masih ada rasa sedih namun sudah lebih tenang"
    ),
}

_DEFAULT_EMOTION_CTX = "sedang merasakan luka emosional akibat putus cinta"


def get_counselor_response(text_original: str, emotion: str, healing_score: float) -> str:
    """
    Hasilkan respons konselor empatik berbahasa Indonesia
    berdasarkan emosi dan kondisi healing pengguna.
    """
    emotion_ctx = EMOTION_CONTEXT.get(emotion, _DEFAULT_EMOTION_CTX)

    # Tentukan nuansa berdasarkan healing_score
    if healing_score < -0.1:
        healing_hint = (
            "Kondisi emosional mereka masih sangat berat dan dalam titik terendah. "
            "Fokus pada validasi perasaan dan memberikan rasa aman — jangan buru-buru "
            "mendorong mereka untuk bangkit."
        )
    elif healing_score < 0.3:
        healing_hint = (
            "Mereka mulai dalam proses pemulihan awal, namun masih rapuh. "
            "Berikan dukungan hangat dan harapan kecil yang realistis."
        )
    else:
        healing_hint = (
            "Mereka sudah cukup stabil secara emosional dan mulai pulih. "
            "Kamu bisa sedikit mendorong mereka untuk melihat ke depan dengan optimisme."
        )

    prompt = f"""Kamu adalah HealMate — konselor digital yang hangat, empatik, dan non-judgmental, 
khusus mendampingi orang yang sedang pulih dari putus cinta.

Konteks pengguna:
- Pesan yang mereka tulis: "{text_original}"
- Emosi yang terdeteksi: {emotion} — artinya mereka {emotion_ctx}.
- {healing_hint}

Tugasmu:
Tulis SATU respons konselor dalam Bahasa Indonesia yang:
1. Dimulai dengan memvalidasi perasaan mereka secara tulus (bukan klise)
2. Menunjukkan bahwa kamu benar-benar mendengar dan memahami
3. Memberikan satu kalimat penyemangat yang hangat dan realistis — BUKAN toxic positivity
4. Terasa seperti pesan dari teman yang bijak, bukan ceramah

Panjang: 3–5 kalimat. Gunakan bahasa yang santai namun penuh perhatian. 
Jangan sebut label emosi secara eksplisit. Jangan gunakan bullet point."""

    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"[Gemini ERROR] {e}")   
        return "Sepertinya kamu sedang menanggung sesuatu yang berat. Aku di sini bersamamu. 💙"


def get_activity_suggestions(emotion: str, healing_score: float) -> list[str]:
    """
    Hasilkan 3–5 saran aktivitas konkret berbahasa Indonesia
    yang sesuai dengan kondisi emosi dan tingkat healing pengguna.
    """
    emotion_ctx = EMOTION_CONTEXT.get(emotion, _DEFAULT_EMOTION_CTX)

    if healing_score < -0.1:
        energy_hint = (
            "Kondisi mereka sangat berat — sarankan aktivitas yang sangat ringan, "
            "tidak membutuhkan energi besar, bisa dilakukan sendiri di rumah."
        )
    elif healing_score < 0.3:
        energy_hint = (
            "Kondisi mereka sedang dalam pemulihan awal — sarankan aktivitas yang "
            "perlahan membangun rutinitas dan koneksi sosial ringan."
        )
    else:
        energy_hint = (
            "Kondisi mereka mulai membaik — sarankan aktivitas yang lebih aktif, "
            "membantu mereka menemukan kembali identitas dan kesenangan hidup."
        )

    prompt = f"""Kamu adalah HealMate — asisten healing untuk orang yang sedang pulih dari putus cinta.

Kondisi pengguna saat ini:
- Emosi: {emotion} — mereka {emotion_ctx}.
- {energy_hint}

Tugasmu:
Berikan tepat 5 saran aktivitas konkret dalam Bahasa Indonesia yang:
1. Spesifik dan bisa langsung dilakukan hari ini
2. Sesuai dengan kondisi emosional dan energi mereka saat ini
3. Beragam: campurkan aktivitas fisik ringan, kreatif, sosial, dan refleksi diri
4. Terasa realistis dan tidak menghakimi

Format output: kembalikan HANYA array JSON berisi 5 string aktivitas.
Contoh format: ["Aktivitas 1", "Aktivitas 2", "Aktivitas 3", "Aktivitas 4", "Aktivitas 5"]
Jangan tambahkan teks apapun di luar array JSON."""

    try:
        import json
        resp = gemini_model.generate_content(prompt)
        raw = resp.text.strip()
        
        raw = raw.replace("```json", "").replace("```", "").strip()
        activities = json.loads(raw)
        if isinstance(activities, list):
            return [str(a) for a in activities[:5]]
        return activities
    except Exception:
        # Fallback aktivitas berdasarkan emosi
        fallback = {
            "anxiety": [
                "Tarik napas dalam 4 hitungan, tahan 4, buang 4 — ulangi 5 kali",
                "Tulis semua kekhawatiranmu di secarik kertas, lalu lipat dan simpan",
                "Berjalan kaki santai 15 menit di sekitar rumah",
                "Seduh teh hangat dan dengarkan playlist lo-fi tanpa HP",
                "Hubungi satu teman yang bikin kamu merasa aman",
            ],
            "anger": [
                "Tulis surat ke mantan — semua yang ingin kamu katakan — lalu jangan dikirim",
                "Olahraga ringan: jumping jack atau jogging di tempat 10 menit",
                "Gambar atau coret-coret bebas di kertas untuk meluapkan energi",
                "Tonton video lucu atau dokumenter yang bikin kamu penasaran",
                "Masak atau pesan makanan favoritmu sebagai self-reward",
            ],
            "acceptance": [
                "Tuliskan 3 hal yang kamu syukuri hari ini",
                "Rapikan satu sudut kamarmu yang berantakan",
                "Mulai baca buku atau podcast yang sudah lama kamu tunda",
                "Rencanakan satu hal kecil yang ingin kamu coba minggu ini",
                "Hubungi teman lama yang sudah lama tidak kamu sapa",
            ],
        }
        return fallback.get(emotion, fallback["anxiety"])


def predict(text: str) -> dict:

    text_in_to_eng = GoogleTranslator(source="auto", target="en").translate(text)
    text_clean = clean_text(text_in_to_eng)

    seq    = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    tensor = tf.constant(padded, dtype=tf.int32)

    output = infer(tensor)

    # Debug: print semua key dan shape-nya
    for key, val in output.items():
        print(f"Key: {key}, Shape: {val.shape}, Sample: {val.numpy()}")

    # Identifikasi output berdasarkan shape, BUKAN urutan key
    # pred_emotion → shape [batch, num_classes] misal (1, 3)
    # pred_healing → shape [batch, 1]           misal (1, 1)
    pred_emotion = None
    pred_healing = None

    for key, val in output.items():
        shape = val.shape
        if len(shape) == 2 and shape[1] > 1:    # [batch, num_classes] → emotion
            pred_emotion = val.numpy()
        elif len(shape) == 2 and shape[1] == 1: # [batch, 1] → healing score
            pred_healing = val.numpy()

    if pred_emotion is None or pred_healing is None:
        raise ValueError(
            f"Tidak bisa identify output. Keys: {list(output.keys())}, "
            f"Shapes: {[str(v.shape) for v in output.values()]}"
        )

    probs         = pred_emotion[0]
    idx           = int(np.argmax(probs))
    emotion       = le.classes_[idx]
    healing_score = round(float(pred_healing[0][0]), 4)

    counselor_response   = get_counselor_response(text, emotion, healing_score)
    activity_suggestions = get_activity_suggestions(emotion, healing_score)

    return {
        "text_original"        : text,
        "text_english"         : text_in_to_eng,
        "text_clean"           : text_clean,
        "emotion"              : emotion,
        "confidence"           : round(float(probs[idx]), 4),
        "all_emotions"         : {
            le.classes_[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
        "healing_score"        : healing_score,
        "counselor_response"   : counselor_response,
        "activity_suggestions" : activity_suggestions,
    }
