import re
import streamlit as st
import pandas as pd
import html as _html
from io import StringIO
from scorer import score_transcript

st.set_page_config(page_title="Communication Scoring Tool", layout="wide")

st.title("🗣️ Communication Skills Scoring Tool")

st.markdown("""
Paste the transcript text or upload a `.txt` file.  
Enter audio duration (seconds) from the recording. Click **Score** to evaluate.
""")

col1, col2 = st.columns([3,1])

with col1:
    transcript = st.text_area("Transcript", height=300, placeholder="Paste transcript here...")
    uploaded = st.file_uploader("Or upload .txt transcript", type=["txt"])
    if uploaded is not None and not transcript.strip():
        transcript = uploaded.getvalue().decode("utf-8")

    words = len(re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", transcript)) if transcript else 0
    sentences = len([s for s in re.split(r'[.!?]+', transcript) if s.strip()]) if transcript else 0
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Word count", f"{words}")
    with c2:
        st.metric("Sentence count", f"{sentences}")

with col2:
    duration = st.number_input("Audio duration (seconds)", min_value=1, value=60, step=1)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    if st.button("Score"):
        if not transcript or len(transcript.strip()) < 3:
            st.error("Please paste or upload a transcript (at least a few words).")
        else:
            with st.spinner("Scoring..."):
                results = score_transcript(transcript, duration)
            st.session_state["last_results"] = results

if "last_results" in st.session_state:
    results = st.session_state["last_results"]
    
    st.subheader("Detailed per-criterion results")
    overall = results.get("overall_score", None)
    if overall is not None:
        st.metric("Overall Score (0-100)", f"{overall:.2f}")
    details = results.get("details", [])
    
    MAPPING = [
        {
            "category": "Content & Structure",
            "items": [
                {"key": "Salutation", "metric": "Salutation Level"},
                {"key": "Mandatory Keywords", "metric": "Keyword Presence (Must have)"},
                {"key": "Optional Keywords", "metric": "Keyword Presence (Good to have)"},
                {"key": "Flow", "metric": "Flow (Salutation -> Name -> Mandatory details -> Optional details -> Closing)"},
            ],
        },
        {"category": "Speech Rate", "items": [{"key": "Speech Rate", "metric": "Speech rate (words/minute)"}]},
        {
            "category": "Language & Grammar",
            "items": [
                {"key": "Grammar", "metric": "Grammar errors count (LanguageTool)"},
                {"key": "Vocabulary (TTR)", "metric": "Vocabulary richness (TTR)"},
            ],
        },
        {"category": "Clarity", "items": [{"key": "Filler Words", "metric": "Filler Word Rate"}]},
        {"category": "Engagement", "items": [{"key": "Sentiment", "metric": "Sentiment / Positivity (transcript-based)"}]},
    ]

    FRAGMENTS = {
        "Salutation": ["salutation"],
        "Mandatory Keywords": ["mandatory"],
        "Optional Keywords": ["optional"],
        "Flow": ["flow"],
        "Speech Rate": ["speech rate"],
        "Grammar": ["grammar"],
        "Vocabulary (TTR)": ["vocabulary", "ttr"],
        "Filler Words": ["filler"],
        "Sentiment": ["sentiment"],
    }

    def find_result(key):
        frags = FRAGMENTS.get(key, [key.lower()])
        for d in details:
            crit = d.get("criterion", "").lower()
            if any(f in crit for f in frags):
                return d
        return None

    table_rows = []
    for group in MAPPING:
        cat = group["category"]
        for it in group["items"]:
            key = it["key"]
            metric_label = it["metric"]
            d = find_result(key)
            score_attr = d.get("score", 0) if d else 0
            scoring_criteria = d.get("feedback", "") if d else ""
            keywords_signals = ""
            if isinstance(scoring_criteria, str) and scoring_criteria.lower().startswith("found:"):
                keywords_signals = scoring_criteria[6:].strip()
            else:
                keywords_signals = str(scoring_criteria)

            score_attr = d.get("score", None) if d else None

            table_rows.append({
                "Criteria": cat,
                "Metric": metric_label,
                "Keywords / Signals": keywords_signals,
                "Score Attributed": score_attr,
                "Weightage": None,
            })

    df = pd.DataFrame(table_rows)

    
    df = df[["Criteria", "Metric", "Keywords / Signals",
             "Score Attributed", "Weightage"]]
    weights = [5, 20, 10, 5, 10, 10, 10, 15, 15]
    for i, w in enumerate(weights):
        if i < len(df):
            df.at[i, "Weightage"] = w

    def render_merged_table(df, criteria_col="Criteria"):
        cols = list(df.columns)
        last = None
        counts = {}
        order = []
        for v in df[criteria_col].tolist():
            if v != last:
                counts[v] = 1
                order.append(v)
                last = v
            else:
                counts[v] += 1

        html = ['<table style="border-collapse:collapse; width:100%;">']
        html.append('<thead><tr>')
        for c in cols:
            html.append(f'<th style="border:1px solid #ddd; padding:8px; text-align:left; background:#333; color:#fff;">{_html.escape(str(c))}</th>')
        html.append('</tr></thead>')
        html.append('<tbody>')

        i = 0
        while i < len(df):
            row = df.iloc[i]
            crit = row[criteria_col]
            span = counts.get(crit, 1)
            html.append('<tr>')
            html.append(f'<td rowspan="{span}" style="border:1px solid #ddd; padding:8px; vertical-align:top;">{_html.escape(str(crit))}</td>')
            for c in cols:
                if c == criteria_col:
                    continue
                html.append(f'<td style="border:1px solid #ddd; padding:8px; vertical-align:top;">{_html.escape(str(row[c]))}</td>')
            html.append('</tr>')
            for j in range(1, span):
                i2 = i + j
                if i2 >= len(df):
                    break
                row2 = df.iloc[i2]
                html.append('<tr>')
                for c in cols:
                    if c == criteria_col:
                        continue
                    html.append(f'<td style="border:1px solid #ddd; padding:8px; vertical-align:top;">{_html.escape(str(row2[c]))}</td>')
                html.append('</tr>')
            i += span

        total_score = 0.0
        total_weight = 0.0
        for _, r in df.iterrows():
            try:
                s = r.get("Score Attributed", 0)
                if s is None or s == "":
                    s_val = 0.0
                else:
                    s_val = float(s)
            except Exception:
                s_val = 0.0
            total_score += s_val
            try:
                w = r.get("Weightage", 0)
                if w is None or w == "":
                    w_val = 0.0
                else:
                    w_val = float(w)
            except Exception:
                w_val = 0.0
            total_weight += w_val

        def fmt_num(x):
            if abs(x - int(x)) < 1e-9:
                return str(int(x))
            return f"{x:.2f}"
        html.append('<tfoot>')
        html.append('<tr>')
        html.append(f'<td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:right; font-weight:bold; background:#333; color:#fff;">Totals</td>')
        html.append(f'<td style="border:1px solid #ddd; padding:8px; text-align:left; font-weight:bold; background:#333; color:#fff;">{_html.escape(fmt_num(total_score))}</td>')
        html.append(f'<td style="border:1px solid #ddd; padding:8px; text-align:left; font-weight:bold; background:#333; color:#fff;">{_html.escape(fmt_num(total_weight))}</td>')
        html.append('</tr>')
        html.append('</tfoot>')
        html.append('</table>')
        return "\n".join(html)

    merged_html = render_merged_table(df, criteria_col="Criteria")
    st.markdown(merged_html, unsafe_allow_html=True)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode("utf-8")
    st.download_button("Download CSV", csv_data, file_name="scoring_table.csv", mime="text/csv")