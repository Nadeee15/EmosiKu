import torch, sys, re
sys.stdout.reconfigure(encoding='utf-8')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import AutoTokenizer, AutoModelForSequenceClassification

stopword = StopWordRemoverFactory().create_stop_word_remover()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text, flags=re.MULTILINE).lower()
    return re.sub(r'\s+', ' ', stopword.remove(text)).strip()

test_texts = [
    "aku bahagia",
    "aku sangat senang hari ini",
    "hidup ini indah",
    "aku ingin mati",
    "aku depresi dan sedih",
    "aku tidak mau hidup lagi",
    "aku cemas dan takut setiap hari",
]

print("=== HASIL PREPROCESSING ===")
for text in test_texts:
    cleaned = clean_text(text)
    print(f"  '{text}' --> '{cleaned}'")

print("\n=== PREDIKSI TANPA clean_text (raw text) ===")
MODEL_NAME = "Nadeee15/EmosiKu-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

print(f"\n{'Teks':<45} | {'Pred':>4} | {'Prob[0]':>8} | {'Prob[1]':>8}")
print("-" * 75)

for text in test_texts:
    # Test with RAW text (no preprocessing)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred = torch.argmax(out.logits, dim=-1).item()
    print(f"{text:<45} | {pred:>4} | {probs[0]:.4f}   | {probs[1]:.4f}")

print("\n=== PREDIKSI DENGAN clean_text ===")
print(f"\n{'Cleaned':<45} | {'Pred':>4} | {'Prob[0]':>8} | {'Prob[1]':>8}")
print("-" * 75)

for text in test_texts:
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred = torch.argmax(out.logits, dim=-1).item()
    print(f"{cleaned:<45} | {pred:>4} | {probs[0]:.4f}   | {probs[1]:.4f}")
