import json
import nltk
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import language_tool_python

nltk.download("punkt")

tool = language_tool_python.LanguageTool("en-US")
sentiment = SentimentIntensityAnalyzer()
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load rubric
with open("rubric.json") as f:
    RUBRIC = json.load(f)


def ttr_score(text):
    words = word_tokenize(text.lower())
    if len(words) == 0:
        return 0
    distinct = len(set(words))
    return distinct / len(words)


def grammar_score(text):
    matches = tool.check(text)
    errors = len(matches)
    words = len(word_tokenize(text))
    errors_per_100 = (errors / words) * 100 if words > 0 else 0
    normalized = 1 - min(errors_per_100 / 10, 1)

    score_map = RUBRIC["grammar"]["score_map"]
    for rule in score_map:
        if "min" in rule and normalized >= rule["min"]:
            if "max" not in rule or normalized <= rule["max"]:
                return rule["score"], normalized
    return 2, normalized


def filler_word_score(text):
    words = word_tokenize(text.lower())
    total = len(words)
    fillers = RUBRIC["filler_words"]["list"]

    count = sum(1 for w in words if w in fillers)
    rate = (count / total) * 100 if total > 0 else 0

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


def sentiment_score(text):
    score = sentiment.polarity_scores(text)["pos"]
    for rule in RUBRIC["sentiment"]["score_map"]:
        if "min" in rule and score >= rule["min"]:
            if "max" not in rule or score <= rule["max"]:
                return rule["score"], score
    return 3, score


def keyword_score(text, key_list):
    text_low = text.lower()
    found = [k for k in key_list if k.lower() in text_low]
    return len(found), found


def salutation_score(text):
    t = text.lower()
    for level, phrases in RUBRIC["salutation"]["keywords"].items():
        for p in phrases:
            if p in t:
                return RUBRIC["salutation"]["levels"][level], level
    return 0, "none"


def speech_rate_score(words, duration):
    wpm = (words / duration) * 60
    ranges = RUBRIC["speech_rate"]["ranges"]

    if wpm > 161:
        return ranges["too_fast"]["score"], wpm
    if 141 <= wpm <= 160:
        return ranges["fast"]["score"], wpm
    if 111 <= wpm <= 140:
        return ranges["ideal"]["score"], wpm
    if 81 <= wpm <= 110:
        return ranges["slow"]["score"], wpm
    return ranges["too_slow"]["score"], wpm


def semantic_similarity(text, rubric_desc="self introduction student basic details family"):
    emb1 = model.encode([text])
    emb2 = model.encode([rubric_desc])
    return float(cosine_similarity([emb1[0]], [emb2[0]])[0][0])


def score_transcript(text, duration):
    words = len(word_tokenize(text))

    results = []

    # Salutation
    s_score, s_level = salutation_score(text)
    results.append({
        "criterion": "Salutation",
        "score": s_score,
        "feedback": f"Detected salutation level: {s_level}"
    })

    # Mandatory keywords
    mand_keywords = RUBRIC["mandatory_keywords"]["keywords"]
    mand_found_count, mand_found = keyword_score(text, mand_keywords)
    mand_score = (mand_found_count / len(mand_keywords)) * RUBRIC["mandatory_keywords"]["weight"]
    results.append({
        "criterion": "Mandatory Keywords",
        "score": mand_score,
        "feedback": f"Found: {mand_found}"
    })

    # Optional keywords
    opt_keywords = RUBRIC["optional_keywords"]["keywords"]
    opt_found_count, opt_found = keyword_score(text, opt_keywords)
    opt_score = (opt_found_count * 2)  # each optional = 2
    results.append({
        "criterion": "Optional Keywords",
        "score": opt_score,
        "feedback": f"Found: {opt_found}"
    })

    # Flow
    flow_score = 5 if "thank you" in text.lower() else 2
    results.append({
        "criterion": "Flow",
        "score": flow_score,
        "feedback": "Basic flow detected." if flow_score == 5 else "Flow partially missing."
    })

    # Speech rate
    sp_score, wpm = speech_rate_score(words, duration)
    results.append({
        "criterion": "Speech Rate",
        "score": sp_score,
        "feedback": f"WPM: {wpm}"
    })

    # Grammar
    g_score, g_norm = grammar_score(text)
    results.append({
        "criterion": "Grammar",
        "score": g_score,
        "feedback": f"Grammar score normalized: {g_norm:.2f}"
    })

    # Vocabulary
    ttr = ttr_score(text)
    vocab_score = next(
        (rule["score"] for rule in RUBRIC["vocabulary"]["score_map"] if "min" in rule and ttr >= rule["min"]),
        2
    )
    results.append({
        "criterion": "Vocabulary (TTR)",
        "score": vocab_score,
        "feedback": f"TTR: {ttr:.2f}"
    })

    # Filler words
    f_score, f_rate = filler_word_score(text)
    results.append({
        "criterion": "Filler Words",
        "score": f_score,
        "feedback": f"Rate: {f_rate:.2f}%"
    })

    # Sentiment
    sent_score, sent_val = sentiment_score(text)
    results.append({
        "criterion": "Sentiment",
        "score": sent_score,
        "feedback": f"Positive sentiment: {sent_val:.2f}"
    })

    # Compute final normalized score
    total_score = sum(r["score"] for r in results)
    final_score = (total_score / 100) * 100

    return {
        "overall_score": final_score,
        "details": results
    }