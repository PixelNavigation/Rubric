# Nirmaan

Here is the requested documentation, including the README with run instructions and scoring details, and the local deployment guide.

-----

# Communication Scoring Tool Documentation

## README.md

### 🗣️ Communication Skills Scoring Tool

This is a **Streamlit application** that evaluates communication skills from a transcribed speech sample against a pre-defined **rubric**. It uses advanced Natural Language Processing (NLP) techniques, including **spaCy**, **NLTK**, **LanguageTool**, and **Sentence-Transformers**, to analyze several factors like grammar, vocabulary, content structure, speech rate, and sentiment.

### 🚀 Running the App Locally

To run this application, you must have **Python 3.8+** and **Java 8+** installed.

#### 1\. Setup Project Files

Ensure you have the following files in your project root directory:

  * `app.py` (The Streamlit frontend code)
  * `scorer.py` (The core scoring logic)
  * `rubric.json` (A file defining the scoring parameters, keywords, and weights - *This file must be present for the app to run*)
  * `requirements.txt` (List of Python dependencies)

#### 2\. Install Dependencies

Create a Python virtual environment and install the required packages.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

#### 3\. Run the Streamlit App

Launch the application from your terminal:

```bash
streamlit run app.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

### ⚖️ Description of Scoring Formula

The application calculates an **Overall Score (0-100)** by summing the scores attributed to **9 distinct criteria**. Each criterion has a predefined weight (defined in the `rubric.json` file and shown in the final table).

The score for each criterion is calculated and normalized using specialized NLP models:

| Criteria Category | Metric | Calculation Basis | Key NLP Tool Used |
| :--- | :--- | :--- | :--- |
| **Content & Structure** | **Salutation** | Based on the presence of predefined salutation keywords (e.g., "Hello," "Good morning"). | Regex/Keyword Matching |
| **Content & Structure** | **Mandatory & Optional Keywords** | Checks for the presence of required personal details (name, age, hobbies) using **spaCy Entity Detection**, **Semantic Similarity (Sentence-Transformers)**, and **Regex**. | spaCy, Sentence-Transformers |
| **Content & Structure** | **Flow** | Scores based on the presence of a logical sequence of elements (Salutation $\rightarrow$ Name $\rightarrow$ Mandatory Details $\rightarrow$ Closing). | Boolean/Sequence Logic |
| **Speech Rate** | **Speech Rate (WPM)** | Calculated as $\text{Word Count} / (\text{Duration in seconds} / 60)$. Score is mapped to ranges (e.g., Ideal: 111-140 WPM). | Arithmetic |
| **Language & Grammar** | **Grammar** | Counts the number of genuine errors (Grammar, Tense, Spelling) detected by the service and normalizes the score based on errors per 100 words. | LanguageTool |
| **Language & Grammar** | **Vocabulary (TTR)** | Uses the **Type-Token Ratio (TTR)**: $\text{Unique Words} / \text{Total Words}$. A higher TTR indicates richer vocabulary. | NLTK (Word Tokenization) |
| **Clarity** | **Filler Words** | Calculates the percentage of filler words (e.g., "um," "like," "you know") relative to the total word count. | Regex/Keyword Matching |
| **Engagement** | **Sentiment** | Measures the positivity and emotional tone of the text using a compound sentiment score, normalized to a 0-1 range. | VADER Sentiment |

-----

## Local Deployment Guide (Detailed Steps)

This guide provides the necessary steps and configuration files to ensure the application, with its complex dependencies, can be reliably deployed and run on any local environment or cloud platform (like Streamlit Community Cloud).

### 1\. Project Structure

Ensure your project directory is organized as follows:

```
/Your-Project-Folder/
├── app.py
├── scorer.py
├── rubric.json  (Must contain scoring rules, weights, and keywords)
├── requirements.txt
├── packages.txt  (System dependencies for Linux/Cloud deployment)
├── .streamlit/
│   └── secrets.toml (Configuration for environment variables)
└── .gitignore
```

### 2\. Configuration Files

These files are essential for installing the required components.

#### A. `requirements.txt` (Python Packages)

List all Python libraries, ensuring the heavy models are installed correctly. Note the inclusion of the specific spaCy model URL for reliable deployment.

```text
streamlit
nltk
vaderSentiment
sentence-transformers
language-tool-python
spacy
# Using 'en_core_web_md' as specified in scorer.py (change to 'sm' for memory optimization)
https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1.tar.gz
```

#### B. `packages.txt` (System Dependencies)

The `language-tool-python` library requires the Java Runtime Environment (JRE). This file ensures the OS package is installed on Linux-based environments (like Streamlit Cloud).

```text
openjdk-17-jre
```

#### C. `.streamlit/secrets.toml` (Java Path Configuration)

This is crucial for the `language-tool-python` library to find the Java executable in shared environments, preventing a `ModuleNotFoundError`. Create the `.streamlit` folder and this file inside it.

```toml
# secrets.toml
# This explicitly tells the language-tool-python library where to find Java.
JAVA_EXECUTABLE_PATH = "/usr/bin/java"
```

### 3\. Execution Steps

1.  **Install Java:** Ensure Java 8 or newer is installed and accessible in your system's PATH. (On cloud platforms, `packages.txt` handles this).
2.  **Install Python Dependencies:** Use the command provided in the README: `pip install -r requirements.txt`.
3.  **Run Streamlit:** The Streamlit command starts the web server, which then runs `app.py`. The initialization code in `scorer.py` (including NLTK downloads, Java server startup, and model loading via `@st.cache_resource`) will execute only once per deployment.
    ```bash
    streamlit run app.py
    ```
