import re
import pandas as pd

# 1. DICTIONARY & REGEX SETUP (Global Variables)
CONTRACTIONS_DICT = {
    # I (dengan apostrof)
    "i'm"       : "i am",
    "i've"      : "i have",
    "i'll"      : "i will",
    "i'd"       : "i would",
    # I (tanpa apostrof — typo/informal)
    "im"        : "i am",
    "ive"       : "i have",
    #"ill"       : "i will",   # hati-hati: "ill" juga berarti sakit
    #"id"        : "i would",  # hati-hati: "id" juga bisa noun
    # you
    "you're"    : "you are",
    "you've"    : "you have",
    "you'll"    : "you will",
    "you'd"     : "you would",
    "youre"     : "you are",
    "youve"     : "you have",
    "youll"     : "you will",
    "youd"      : "you would",
    # he
    "he's"      : "he is",
    "he'll"     : "he will",
    "he'd"      : "he would",
    "hes"       : "he is",
    "hed"       : "he would",
    # she
    "she's"     : "she is",
    "she'll"    : "she will",
    "she'd"     : "she would",
    "shes"      : "she is",
    "shed"      : "she would",
    # it
    "it's"      : "it is",
    "it'll"     : "it will",
    "it'd"      : "it would",
    #"its"       : "it is",    # hati-hati: "its" juga bisa possessive
    # we
    "we're"     : "we are",
    "we've"     : "we have",
    "we'll"     : "we will",
    "we'd"      : "we would",
    #"were"      : "we are",   # hati-hati: "were" juga past tense "be"
    "weve"      : "we have",
    "wed"       : "we would",
    # they
    "they're"   : "they are",
    "they've"   : "they have",
    "they'll"   : "they will",
    "they'd"    : "they would",
    "theyre"    : "they are",
    "theyve"    : "they have",
    "theyll"    : "they will",
    "theyd"     : "they would",
    # negations (dengan apostrof)
    "aren't"    : "are not",
    "isn't"     : "is not",
    "wasn't"    : "was not",
    "weren't"   : "were not",
    "don't"     : "do not",
    "doesn't"   : "does not",
    "didn't"    : "did not",
    "won't"     : "will not",
    "wouldn't"  : "would not",
    "can't"     : "can not",
    "cannot"    : "can not",
    "couldn't"  : "could not",
    "shouldn't" : "should not",
    "hadn't"    : "had not",
    "hasn't"    : "has not",
    "haven't"   : "have not",
    "mustn't"   : "must not",
    "needn't"   : "need not",
    "daren't"   : "dare not",
    "shan't"    : "shall not",
    # negations (tanpa apostrof — typo/informal)
    "arent"     : "are not",
    "isnt"      : "is not",
    "wasnt"     : "was not",
    "werent"    : "were not",
    "dont"      : "do not",
    "doesnt"    : "does not",
    "didnt"     : "did not",
    "wont"      : "will not",
    "wouldnt"   : "would not",
    "cant"      : "can not",
    "couldnt"   : "could not",
    "shouldnt"  : "should not",
    "hadnt"     : "had not",
    "hasnt"     : "has not",
    "havent"    : "have not",
    "mustnt"    : "must not",
    # have/would combinations (dengan apostrof)
    "that've"   : "that have",
    "who've"    : "who have",
    "would've"  : "would have",
    "could've"  : "could have",
    "should've" : "should have",
    "might've"  : "might have",
    "must've"   : "must have",
    # have/would combinations (tanpa apostrof)
    "wouldve"   : "would have",
    "couldve"   : "could have",
    "shouldve"  : "should have",
    "mightve"   : "might have",
    "mustve"    : "must have",
    # misc (dengan apostrof)
    "that's"    : "that is",
    "that'd"    : "that would",
    "there's"   : "there is",
    "there're"  : "there are",
    "there'll"  : "there will",
    "who's"     : "who is",
    "who'd"     : "who would",
    "who'll"    : "who will",
    "what's"    : "what is",
    "what're"   : "what are",
    "what'll"   : "what will",
    "what'd"    : "what did",
    "where's"   : "where is",
    "where'd"   : "where did",
    "when's"    : "when is",
    "why's"     : "why is",
    "how's"     : "how is",
    "how'd"     : "how did",
    "how'll"    : "how will",
    "let's"     : "let us",
    "y'all"     : "you all",
    # misc (tanpa apostrof)
    "thats"     : "that is",
    "theres"    : "there is",
    "whos"      : "who is",
    "whats"     : "what is",
    "wheres"    : "where is",
    "whens"     : "when is",
    "whys"      : "why is",
    "hows"      : "how is",
    "lets"      : "let us",
    "yall"      : "you all",
    # informal/slang
    "gonna"     : "going to",
    "wanna"     : "want to",
    "gotta"     : "got to",
    "kinda"     : "kind of",
    "sorta"     : "sort of",
    "dunno"     : "do not know",
    "ain't"     : "is not",
    "aint"      : "is not",
    "tryna"     : "trying to",
    "hafta"     : "have to",
    "oughta"    : "ought to",
    "supposta"  : "supposed to",
    "useta"     : "used to",
}

# Compile regex saat modul di-import agar cepat
CONTRACTIONS_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS_DICT.keys()) + r')\b',
    re.IGNORECASE
)

# 2. CORE FUNCTIONS (Siap di-import ke Notebook lain)
def expand_contractions(text):
    """Fungsi helper untuk menjabarkan singkatan bahasa Inggris."""
    def replace(match):
        token = match.group(0).lower()
        return CONTRACTIONS_DICT.get(token, token)
    return CONTRACTIONS_PATTERN.sub(replace, str(text))

def clean_text(text):
    """
    Fungsi utama untuk membersihkan teks.
    Penggunaan di notebook lain:
    from preprocessing import clean_text
    df['clean_text'] = df['text'].apply(clean_text)
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # hapus URL
    text = expand_contractions(text)             # expand contractions
    text = re.sub(r"[^\w\s]", " ", text)         # hapus tanda baca
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # hapus non-huruf & angka
    text = re.sub(r"\s+", " ", text).strip()     # rapikan spasi
    return text

# 3. DATASET PIPELINE (Opsional / Helper untuk Batch)
def assess_data(df):
    """Mencetak info duplikat dan missing values."""
    print("\n" + "=" * 50)
    print("ASSESSING DATA")
    print("=" * 50)

    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        "jumlah_missing": missing,
        "persentase (%)": (missing / len(df) * 100).round(2)
    })
    print("\n[ Missing Value ]")
    print(missing_df)

    if "text" in df.columns:
        n_dup = df.duplicated(subset=["text"]).sum()
        print(f"\n[ Duplikat ]\nJumlah baris : {n_dup:,}\nPersentase   : {n_dup / len(df) * 100:.2f}%")

def clean_data(df):
    """Membersihkan seluruh dataframe menggunakan clean_text."""
    df_clean = df.dropna(subset=["text"]).copy()
    df_clean = df_clean.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_clean["text_clean"] = df_clean["text"].apply(clean_text)
    return df_clean

# 4. EXECUTABLE BLOCK (Hanya jalan jika file ini di-run langsung)
if __name__ == "__main__":
    # Bagian ini TIDAK AKAN JALAN jika kamu cuma melakukan `import preprocessing` di notebook lain.
    # Bagian ini hanya jalan jika kamu mengeksekusi file ini langsung di terminal (python preprocessing.py)
    
    input_file = "nama_dataset_mentah_kamu.csv" 
    
    try:
        print(f"Membaca data dari '{input_file}'...")
        df_raw = pd.read_csv(input_file)
        
        assess_data(df_raw)
        df_clean = clean_data(df_raw)

        output_cols = ["text_clean"]
        for col in ["label", "emotion", "source"]:
            if col in df_clean.columns:
                output_cols.append(col)
                
        output_file = "dataset_preprocessed.csv"
        df_clean[output_cols].to_csv(output_file, index=False)

        print(f"\nSelesai! Data disimpan di '{output_file}' dengan {len(df_clean):,} baris.")
        
    except FileNotFoundError:
        print(f"\n[ERROR] File '{input_file}' tidak ditemukan.")
