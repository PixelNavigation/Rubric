import json
import nltk
import re
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import spacy

nltk.download("punkt")

tool = language_tool_python.LanguageToolPublicAPI("en-US")
sentiment = SentimentIntensityAnalyzer()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_md")

with open("rubric.json") as f:
    RUBRIC = json.load(f)


# NAME DETECTION (spaCy + regex)
def detect_name_spacy(text):
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    if names:
        return True, names[0], names

    regex_patterns = [
        r"my name is\s+([A-Za-z]+)",
        r"i am\s+([A-Za-z]+)",
        r"i'm\s+([A-Za-z]+)",
        r"myself\s+([A-Za-z]+)",
        r"this is\s+([A-Za-z]+)"
    ]

    for p in regex_patterns:
        m = re.search(p, text.lower())
        if m:
            name = m.group(1).capitalize()
            return True, name, [name]

    return False, None, []


# TYPE-TOKEN RATIO
def ttr_score(text):
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)


# GRAMMAR SCORE
def grammar_score(text):
    matches = tool.check(text)

    true_errors = 0

    grammar_categories = [
        "Grammar",
        "Tense",
        "Agreement",
        "Confused Words",
        "Miscellaneous",
        "Possible Typo",
        "Spelling"
    ]

    grammar_rule_keywords = [
        "GRAMMAR",
        "TENSE",
        "AGREEMENT",
        "WORD_CHOICE",
        "CONFUSED_WORDS",
        "TYPO",
        "SPELLING"
    ]

    for m in matches:
        if hasattr(m.category, "name"):
            category_name = m.category.name
        else:
            category_name = str(m.category) if m.category else ""

        message = m.message.lower()

        if any(cat.lower() in category_name.lower() for cat in grammar_categories):
            true_errors += 1
            continue

        rule_id = str(getattr(m, "ruleId", "")).lower()
        if any(key.lower() in rule_id for key in grammar_rule_keywords):
            true_errors += 1
            continue

        if "spelling" in message or "misspelling" in message or "typo" in message:
            true_errors += 1
            continue

    # Word count
    words = len(word_tokenize(text))
    if words == 0:
        return 0, 0

    # Errors per 100 words
    errors_per_100 = (true_errors / words) * 100

    # Rubric normalization
    normalized = 1 - min(errors_per_100 / 10, 1)

    # Map to rubric
    for rule in RUBRIC["grammar"]["score_map"]:
        if normalized >= rule["min"] and ("max" not in rule or normalized <= rule["max"]):
            return rule["score"], normalized

    return 2, normalized




# FILLER WORD SCORE
def filler_word_score(text):
    fillers = RUBRIC["filler_words"]["list"]
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text.lower())

    total = len(tokens)
    if total == 0:
        return 15, 0

    count = sum(1 for w in tokens if w in fillers)
    rate = (count / total) * 100

    if rate <= 3: return 15, rate
    if rate <= 6: return 12, rate
    if rate <= 9: return 9, rate
    if rate <= 12: return 6, rate
    return 3, rate


# SENTIMENT SCORE
def sentiment_score(text):
    vs = sentiment.polarity_scores(text)
    val = (vs["compound"] * 0.7) + (vs["pos"] * 0.3)
    val = max(0, min(val, 1))

    for rule in RUBRIC["sentiment"]["score_map"]:
        if val >= rule["min"] and ("max" not in rule or val <= rule["max"]):
            return rule["score"], val

    return 3, val


# SALUTATION SCORE
def salutation_score(text):
    t = text.lower()

    for level in ["excellent", "good", "normal"]:
        for phrase in RUBRIC["salutation"]["keywords"][level]:
            if re.search(r"\b" + re.escape(phrase) + r"\b", t):
                return RUBRIC["salutation"]["levels"][level], level

    return 0, "none"


# SPEECH RATE SCORE
def speech_rate_score(word_count, duration):
    if duration == 0:
        return 0, 0

    wpm = (word_count / duration) * 60
    r = RUBRIC["speech_rate"]["ranges"]

    if wpm >= 161: return r["too_fast"]["score"], wpm
    if 141 <= wpm <= 160: return r["fast"]["score"], wpm
    if 111 <= wpm <= 140: return r["ideal"]["score"], wpm
    if 81 <= wpm <= 110: return r["slow"]["score"], wpm
    return r["too_slow"]["score"], wpm


# SEMANTIC HOBBY DETECTION
hobby_phrases_base = [
    "playing cricket", "reading books", "playing football", "listening to music",
    "singing", "dancing", "painting", "drawing", "coding", "watching movies"
]
hobby_embeddings = embed_model.encode(hobby_phrases_base, convert_to_tensor=True)

def semantic_hobby_detect(text):
    doc = nlp(text)
    phrases = []

    for chunk in doc.noun_chunks:
        emb = embed_model.encode(chunk.text, convert_to_tensor=True)
        sim = util.cos_sim(emb, hobby_embeddings).max().item()
        if sim > 0.55:
            phrases.append(chunk.text)

    return len(phrases) > 0, phrases


# KEYWORD PRESENCE SCORE (MANDATORY + OPTIONAL)
def keyword_presence_score(text):
    t = text.lower()

    M = RUBRIC["keywords"]["mandatory"]
    O = RUBRIC["keywords"]["optional"]

    mandatory_found = []
    optional_found = []

    # ---- Mandatory 1: Name
    name_present, name_val, names_all = detect_name_spacy(text)
    if name_present:
        mandatory_found.append("name")

    # ---- Mandatory 2: Age
    if re.search(r"\b\d{1,2}\s*years? old\b", t) or "age" in t:
        mandatory_found.append("age")

    # ---- Mandatory 3: Class/School
    if any(x in t for x in ["class", "grade", "standard", "school"]):
        mandatory_found.append("class/school")

    # ---- Mandatory 4: Family
    if any(x in t for x in ["family", "parents", "mother", "father", "sister", "brother"]):
        mandatory_found.append("family")

    # ---- Mandatory 5: Hobbies/Interests
    hobby_regex = [
        r"i like to",
        r"i enjoy",
        r"my hobby",
        r"my hobbies",
        r"i love to",
        r"interested in",
        r"my interests include"
    ]
    hobby_flag = any(re.search(r, t) for r in hobby_regex)

    semantic_flag, semantic_phrases = semantic_hobby_detect(text)

    if hobby_flag or semantic_flag:
        mandatory_found.append("hobbies/interests")

    # ---- Optional keywords
    for opt in O:
        if opt in t:
            optional_found.append(opt)

    # ---- Scores
    mand_score = len(set(mandatory_found)) * RUBRIC["keywords"]["mandatory_score_each"]
    opt_score = len(set(optional_found)) * RUBRIC["keywords"]["optional_score_each"]

    mand_score = min(mand_score, RUBRIC["keywords"]["weight_mandatory"])
    opt_score = min(opt_score, RUBRIC["keywords"]["weight_optional"])

    return {
        "mandatory_score": mand_score,
        "optional_score": opt_score,
        "mandatory_found": list(set(mandatory_found)),
        "optional_found": list(set(optional_found)),
        "hobby_examples": semantic_phrases,
        "names_detected": names_all
    }


# FLOW
def flow_score(text):
    t = text.lower()

    # salutation detection
    has_sal = any(
        phrase in t
        for level in RUBRIC["salutation"]["keywords"].values()
        for phrase in level
    )

    # name detection
    has_name, _, _ = detect_name_spacy(text)

    # mandatory detection (keyword function returns dict)
    k = keyword_presence_score(text)
    has_mandatory = k["mandatory_score"] > 0

    # optional detection
    has_optional = k["optional_score"] > 0

    # closing detection
    has_close = "thank you" in t or "thanks" in t

    score = 0
    if has_sal: score += 1
    if has_name: score += 1
    if has_mandatory: score += 1
    if has_optional: score += 1
    if has_close: score += 1

    return min(score, 5), {
        "salutation": has_sal,
        "name": has_name,
        "mandatory_present": has_mandatory,
        "optional_present": has_optional,
        "closing": has_close
    }



# MAIN PIPELINE
def clean_word_count(text):
    return len(re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text))


def score_transcript(text, duration):
    words = clean_word_count(text)

    results = []

    # SALUTATION
    sal_score, sal_feedback = salutation_score(text)
    results.append({"criterion": "Salutation", "score": sal_score, "feedback": sal_feedback})

    # KEYWORDS (mandatory + optional)
    k = keyword_presence_score(text)

    results.append({
        "criterion": "Mandatory Keywords",
        "score": k["mandatory_score"],
        "feedback": {
            "found": k["mandatory_found"],
            "hobbies": k["hobby_examples"],
            "names": k["names_detected"]
        }
    })

    results.append({
        "criterion": "Optional Keywords",
        "score": k["optional_score"],
        "feedback": {
            "found": k["optional_found"]
        }
    })

    # FLOW
    f_score, f_feedback = flow_score(text)
    results.append({"criterion": "Flow", "score": f_score, "feedback": f_feedback})

    # SPEECH RATE
    sr_score, wpm = speech_rate_score(words, duration)
    results.append({"criterion": "Speech Rate", "score": sr_score, "feedback": f"{wpm:.2f} wpm"})

    # GRAMMAR
    g_score, g_norm = grammar_score(text)
    results.append({"criterion": "Grammar", "score": g_score, "feedback": f"Normalized: {g_norm:.2f}"})

    # VOCABULARY
    ttr = ttr_score(text)
    vocab_score = next((rule["score"] for rule in RUBRIC["vocabulary"]["score_map"]
                        if ttr >= rule["min"]), 2)
    results.append({"criterion": "Vocabulary (TTR)", "score": vocab_score, "feedback": f"TTR: {ttr:.2f}"})

    # FILLER WORDS
    fw_score, fw_rate = filler_word_score(text)
    results.append({"criterion": "Filler Words", "score": fw_score, "feedback": f"{fw_rate:.2f}%"})

    # SENTIMENT
    s_score, s_val = sentiment_score(text)
    results.append({"criterion": "Sentiment", "score": s_score, "feedback": f"{s_val:.2f}"})

    # FINAL SCORE
    total_score = sum(r["score"] for r in results)
    return {"overall_score": total_score, "details": results}
