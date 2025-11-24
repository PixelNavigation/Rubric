import json
import nltk
import re
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import language_tool_python
import spacy

# Downloads
nltk.download("punkt")

# Load external tools
tool = language_tool_python.LanguageTool("en-US")
sentiment = SentimentIntensityAnalyzer()
model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_md")

# Load rubric
with open("rubric.json") as f:
    RUBRIC = json.load(f)


# ---------------------------------------------------------
# NAME DETECTION (spaCy NER)
# ---------------------------------------------------------
def detect_name_spacy(text):
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # If spaCy detects, return it
    if names:
        return True, names[0], names

    # Fallback regex patterns
    fallback_patterns = [
        r"my name is\s+([A-Za-z]+)",
        r"i am\s+([A-Za-z]+)",
        r"i'm\s+([A-Za-z]+)",
        r"myself\s+([A-Za-z]+)",
        r"this is\s+([A-Za-z]+)",
        r"([A-Za-z]+)\s+here"
    ]

    t = text.lower()

    for pattern in fallback_patterns:
        match = re.search(pattern, t, re.IGNORECASE)
        if match:
            name = match.group(1).capitalize()
            return True, name, [name]

    return False, None, []



# ---------------------------------------------------------
# TYPE TOKEN RATIO (Vocabulary richness)
# ---------------------------------------------------------
def ttr_score(text):
    words = word_tokenize(text.lower())
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)


# ---------------------------------------------------------
# GRAMMAR SCORING
# ---------------------------------------------------------
def grammar_score(text):
    matches = tool.check(text)
    errors = len(matches)

    # Use clean word counter, not NLTK tokenizer
    words = clean_word_count(text)

    if words == 0:
        return 0, 0

    # Errors per 100 words
    errors_per_100 = (errors / words) * 100

    # Normalize to 0–1 scale
    normalized = 1 - min(errors_per_100 / 10, 1)  # 0 errors = 1, many errors = 0

    # Map to rubric
    score_map = RUBRIC["grammar"]["score_map"]
    for rule in score_map:
        if "min" in rule and normalized >= rule["min"]:
            if "max" not in rule or normalized <= rule["max"]:
                return rule["score"], normalized

    return 2, normalized

# ---------------------------------------------------------
# FILLER WORDS
# ---------------------------------------------------------
def filler_word_score(text):
    fillers = RUBRIC["filler_words"]["list"]

    # Extract clean tokens (no punctuation)
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text.lower())
    
    # Total real words
    total = len(tokens)

    # Count filler words ONLY from clean tokens
    count = sum(1 for w in tokens if w in fillers)

    # Avoid division-by-zero
    rate = (count / total) * 100 if total > 0 else 0

    ranges = RUBRIC["filler_words"]["ranges"]

    if rate <= 3:
        return 15, rate
    elif rate <= 6:
        return 12, rate
    elif rate <= 9:
        return 9, rate
    elif rate <= 12:
        return 6, rate
    else:
        return 3, rate


# ---------------------------------------------------------
# SENTIMENT (VADER)
# ---------------------------------------------------------
def sentiment_score(text):
    vs = sentiment.polarity_scores(text)

    # final sentiment = positive + (compound * 0.5)
    final_sent = vs["pos"] + (vs["compound"] * 0.5)
    final_sent = max(0, min(final_sent, 1))  # clamp 0–1

    # map to rubric
    for rule in RUBRIC["sentiment"]["score_map"]:
        if "min" in rule and final_sent >= rule["min"]:
            if "max" not in rule or final_sent <= rule["max"]:
                return rule["score"], final_sent

    return 3, final_sent



# ---------------------------------------------------------
# SALUTATION SCORING
# ---------------------------------------------------------
def salutation_score(text):
    t = text.lower()

    priority = ["excellent", "good", "normal"]

    for level in priority:
        for phrase in RUBRIC["salutation"]["keywords"][level]:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, t):
                return RUBRIC["salutation"]["levels"][level], level

    return 0, "none"


# ---------------------------------------------------------
# SPEECH RATE
# ---------------------------------------------------------
def speech_rate_score(words, duration):
    if duration == 0:
        return 0, 0  # avoid division error

    wpm = (words / duration) * 60
    ranges = RUBRIC["speech_rate"]["ranges"]

    if wpm >= 161:
        return ranges["too_fast"]["score"], wpm
    elif 141 <= wpm <= 160:
        return ranges["fast"]["score"], wpm
    elif 111 <= wpm <= 140:
        return ranges["ideal"]["score"], wpm
    elif 81 <= wpm <= 110:
        return ranges["slow"]["score"], wpm
    else:  # 80 or less
        return ranges["too_slow"]["score"], wpm



# ---------------------------------------------------------
# KEYWORD SCORING (Mandatory 5*4 + Optional 15*2)
# ---------------------------------------------------------
def keyword_presence_score(text):
    text_low = text.lower()
    mand = RUBRIC["keywords"]["mandatory"]
    opt = RUBRIC["keywords"]["optional"]

    found = []
    score = 0

    # --- NAME detection (spaCy NER) ---
    name_present, name_val, all_names = detect_name_spacy(text)
    if name_present:
        score += RUBRIC["keywords"]["mandatory_score_each"]
        found.append("name")

    # --- Mandatory detection (except name) ---
    for k in mand:
        if k == "name":
            continue
        if k.lower() in text_low:
            score += RUBRIC["keywords"]["mandatory_score_each"]
            found.append(k)

    # --- Optional detection ---
    opt_found = [k for k in opt if k.lower() in text_low]
    score += len(opt_found) * RUBRIC["keywords"]["optional_score_each"]

    found.extend(opt_found)

    # Clamp to 30 max
    score = min(score, RUBRIC["keywords"]["weight"])

    return score, found, all_names


# ---------------------------------------------------------
# FLOW SCORING (Salutation → Name → Mandatory → Optional → Closing)
# ---------------------------------------------------------
def flow_score(text):
    t = text.lower()

    has_sal = any(
        phrase in t
        for level in RUBRIC["salutation"]["keywords"].values()
        for phrase in level
    )
    has_name, _, _ = detect_name_spacy(text)
    has_mand = any(k in t for k in RUBRIC["keywords"]["mandatory"])
    has_opt = any(k in t for k in RUBRIC["keywords"]["optional"])
    has_close = "thank you" in t

    score = 0
    if has_sal: score += 1
    if has_name: score += 1
    if has_mand: score += 1
    if has_opt: score += 1
    if has_close: score += 1

    return min(score, 5), {
        "salutation": has_sal,
        "name": has_name,
        "mandatory": has_mand,
        "optional": has_opt,
        "closing": has_close
    }

def clean_word_count(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text)
    return len(words)


# ---------------------------------------------------------
# MAIN SCORING PIPELINE
# ---------------------------------------------------------
def score_transcript(text, duration):
    words = clean_word_count(text)

    results = []

    # Salutation
    s_score, s_level = salutation_score(text)
    results.append({
        "criterion": "Salutation (5)",
        "score": s_score,
        "feedback": f"Detected salutation level: {s_level}"
    })

    # Keywords (30)
    key_score, found_keys, detected_names = keyword_presence_score(text)
    results.append({
        "criterion": "Keyword Presence (30)",
        "score": key_score,
        "feedback": f"Found: {found_keys}, Names: {detected_names}"
    })

    # Flow (5)
    f_score, flow_info = flow_score(text)
    results.append({
        "criterion": "Flow (5)",
        "score": f_score,
        "feedback": flow_info
    })

    # Speech Rate (10)
    sp_score, wpm = speech_rate_score(words, duration)
    results.append({
        "criterion": "Speech Rate (10)",
        "score": sp_score,
        "feedback": f"{wpm:.2f} WPM"
    })

    # Grammar (10)
    g_score, g_norm = grammar_score(text)
    results.append({
        "criterion": "Grammar (10)",
        "score": g_score,
        "feedback": f"Normalized grammar: {g_norm:.2f}"
    })

    # Vocabulary (10)
    ttr = ttr_score(text)
    vocab_score = next(
        (rule["score"] for rule in RUBRIC["vocabulary"]["score_map"]
         if "min" in rule and ttr >= rule["min"]),
        2
    )
    results.append({
        "criterion": "Vocabulary (10)",
        "score": vocab_score,
        "feedback": f"TTR: {ttr:.2f}"
    })

    # Filler (15)
    f_score, f_rate = filler_word_score(text)
    results.append({
        "criterion": "Filler Words (15)",
        "score": f_score,
        "feedback": f"Rate: {f_rate:.2f}%"
    })

    # Sentiment (15)
    sent_score, sent_val = sentiment_score(text)
    results.append({
        "criterion": "Sentiment (15)",
        "score": sent_score,
        "feedback": f"Positive sentiment: {sent_val:.2f}"
    })

    # Final Score (100)
    total_score = sum(r["score"] for r in results if "(info)" not in r["criterion"])
    final_score = min(total_score, 100)

    return {
        "overall_score": final_score,
        "details": results
    }
