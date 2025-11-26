# Capture The Flag – STU160

This repository contains my solution for the AI-based Capture the Flag challenge.  
The objective is to detect a secretly manipulated book and extract three flags using hashing, text-matching, and explainable machine learning (SHAP).

## Summary of Approach

1. I computed SHA256("STU160") and extracted the first 8 hex characters. This hash is used to locate the injected fake review.
2. Next, I filtered the books dataset to find books with `rating_number = 1234` and `average_rating = 5.0`. These represent potential candidates used in the manipulation.
3. I scanned **all text-based columns** in reviews to detect where the hash appears. The row that contains the hash identifies the intentionally injected fake review.
4. Using the book ID of that review, I located the associated book title, cleaned the first 8 characters, and generated SHA256 → FLAG1.
5. FLAG2 is simply the injected hash inside the fake review.
6. To compute FLAG3, I trained a logistic regression model to classify suspicious vs genuine reviews. I used TF-IDF for text encoding and analyzed only genuine reviews using SHAP values to identify the top 3 words that reduce suspicion. Those words + student number → SHA256 → FLAG3.

All three flags are stored in `flags.txt`, and the full solution is implemented in `solver.py`.
