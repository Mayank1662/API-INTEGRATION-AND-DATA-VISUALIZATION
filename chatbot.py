
"""
AI Chatbot with spaCy — rule-of-thumb similarity matcher.
Run: python chatbot.py
"""

import json, random, pathlib, sys
import spacy

# ---------- Config ----------
MODEL_NAME = "en_core_web_md"           # uses vectors for similarity
THRESHOLD  = 0.75                       # similarity cutoff
INTENTS_F  = pathlib.Path("intents.json")
# -----------------------------

# 1) Load spaCy model (download if missing)
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"Downloading {MODEL_NAME} …")
    from spacy.cli import download
    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)

# 2) Load intents
with INTENTS_F.open(encoding="utf-8") as f:
    intents = json.load(f)

# Pre-vectorise patterns for speed
patterns = []
for intent in intents:
    name = intent["intent"]
    for pat in intent["patterns"]:
        patterns.append((nlp(pat), name))

def detect_intent(text: str) -> str:
    """Return the best-matching intent name or 'fallback'."""
    doc = nlp(text)
    best_name, best_sim = None, 0.0
    for pat_doc, name in patterns:
        sim = doc.similarity(pat_doc)
        if sim > best_sim:
            best_name, best_sim = name, sim
    return best_name if best_sim >= THRESHOLD else "fallback"

def respond(intent_name: str) -> str:
    """Pick a random response for the intent (or default)."""
    for intent in intents:
        if intent["intent"] == intent_name:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

def chat():
    print("🤖  Chatbot is ready!  (type 'quit' to exit)")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            print("Bot: Bye! 👋")
            break
        intent = detect_intent(user)
        print("Bot:", respond(intent))

if __name__ == "__main__":
    chat()

