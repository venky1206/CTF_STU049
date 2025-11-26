import pandas as pd
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import numpy as np

# =========================
# CONFIG
# =========================
STUDENT_ID = "STU160"

def sha256_hex(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()

# Hash used in fake review
TARGET_HASH = sha256_hex(STUDENT_ID)[:8].upper()
print("Your 8-char hash:", TARGET_HASH)

# =========================
# STEP 1: LOAD DATA
# =========================
books = pd.read_csv("books.csv")
reviews = pd.read_csv("reviews.csv")

# --- Make sure we have book_id + title in books ---
if "book_id" not in books.columns:
    if "parent_asin" in books.columns:
        books["book_id"] = books["parent_asin"]
    elif "asin" in books.columns:
        books["book_id"] = books["asin"]
    else:
        raise ValueError("books.csv must contain book_id / parent_asin / asin")

if "title" not in books.columns:
    raise ValueError("books.csv must contain a 'title' column")

books = books[["book_id", "title"]]

# --- Make sure we have book_id + text in reviews ---
if "book_id" not in reviews.columns:
    if "parent_asin" in reviews.columns:
        reviews["book_id"] = reviews["parent_asin"]
    elif "asin" in reviews.columns:
        reviews["book_id"] = reviews["asin"]
    else:
        raise ValueError("reviews.csv must contain book_id / parent_asin / asin")

if "text" not in reviews.columns:
    # if no 'text' column, try to build one from other string columns
    string_cols = [c for c in reviews.columns if reviews[c].dtype == "object"]
    if not string_cols:
        raise ValueError("reviews.csv has no 'text' column and no string columns to join.")
    reviews["text"] = reviews[string_cols].fillna("").agg(" ".join, axis=1)

reviews = reviews[["book_id", "text"]]

print("\nBooks columns:", books.columns.tolist())
print("Reviews columns:", reviews.columns.tolist())
print("Books rows:", len(books), "Reviews rows:", len(reviews))

# =========================
# STEP 2: FIND FAKE REVIEW + BOOK (FLAG1)
# =========================

# Search hash in ALL review texts (case-insensitive)
mask = reviews["text"].astype(str).str.contains(TARGET_HASH, case=False, na=False)
fake_review = reviews[mask]

if fake_review.empty:
    print("\nERROR: Fake review not found. Your current CSVs probably do not contain "
          "the CTF-manipulated data. solver.py will stop here.")
    raise SystemExit(1)

print("\nFake review row(s):")
print(fake_review.head())

book_id = fake_review.iloc[0]["book_id"]

# Match with book
book_row = books[books["book_id"] == book_id]
if book_row.empty:
    raise ValueError("Book for fake review not found in books.csv")

title = book_row.iloc[0]["title"]
clean_title = "".join(title.split())[:8]  # first 8 non-space characters
FLAG1 = sha256_hex(clean_title)

print("\nMatched book_id:", book_id)
print("Book title:", title)
print("Cleaned first 8 chars:", clean_title)
print("FLAG1 =", FLAG1)

# =========================
# STEP 3: FLAG2
# =========================
FLAG2 = f"FLAG2{{{TARGET_HASH}}}"
print("\nFLAG2 =", FLAG2)

# =========================
# STEP 4: MODEL + SHAP FOR FLAG3
# =========================

# All reviews for this book
book_reviews = reviews[reviews["book_id"] == book_id].copy()
print("\nNumber of reviews for this book:", len(book_reviews))

if len(book_reviews) < 5:
    print("Warning: very few reviews; SHAP may be unstable.")

# Label suspicious vs genuine
def label_suspicious(text: str) -> int:
    t = str(text).lower()
    words = t.split()
    short = len(words) < 12
    superlatives = any(w in t for w in ["best", "amazing", "awesome", "incredible", "perfect"])
    return 1 if (short and superlatives) else 0

book_reviews["is_suspicious"] = book_reviews["text"].apply(label_suspicious)

# TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
X = vectorizer.fit_transform(book_reviews["text"])
y = book_reviews["is_suspicious"].values

# If all labels are same, we can't train a classifier; fall back to dummy flags
if len(np.unique(y)) == 1:
    print("\nAll reviews labelled the same; cannot train classifier. "
          "Using placeholder words for FLAG3.")
    top_words = ["story", "characters", "writing"]
else:
    # Train simple classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # SHAP explainer
    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    # Handle possible list output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Genuine reviews (low suspicion)
    genuine_idx = book_reviews[book_reviews["is_suspicious"] == 0].index
    if len(genuine_idx) == 0:
        print("\nNo genuine reviews detected; using all reviews for SHAP mean.")
        genuine_idx = book_reviews.index

    mean_shap = shap_values[genuine_idx].mean(axis=0)

    # 3 most negative words = reduce suspicion
    top_idx = mean_shap.argsort()[:3]
    feature_names = vectorizer.get_feature_names_out()
    top_words = [feature_names[i] for i in top_idx]

print("\nTop 3 words that reduce suspicion:", top_words)

# =========================
# STEP 5: FLAG3
# =========================
student_num = STUDENT_ID.replace("STU", "")
FLAG3_raw = "".join(top_words) + student_num
FLAG3_hash = sha256_hex(FLAG3_raw)[:10]
FLAG3 = f"FLAG3{{{FLAG3_hash}}}"

print("\nFLAG3 raw string:", FLAG3_raw)
print("FLAG3 =", FLAG3)

# =========================
# STEP 6: WRITE flags.txt
# =========================
with open("flags.txt", "w") as f:
    f.write(f"FLAG1 = {FLAG1}\n")
    f.write(f"FLAG2 = {FLAG2}\n")
    f.write(f"FLAG3 = {FLAG3}\n")

print("\nflags.txt written successfully.")
