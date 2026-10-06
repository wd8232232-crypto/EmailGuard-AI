# ==========================================
# PHISHING EMAIL DETECTION MODEL
# Using Python + Scikit-learn
# ==========================================

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. SAMPLE DATASET
# ==========================================

data = {
    "email": [
        "Congratulations! You won a free prize. Click here to claim your reward.",
        "Your account has been suspended. Verify your password immediately.",
        "URGENT! Your bank account will be closed. Click this link now.",
        "You have won $1000. Send your bank details to receive the money.",
        "Click http://secure-login-example.com and verify your account.",
        "Your password expires today. Login immediately to avoid losing access.",
        "You are selected for a free gift. Click the link to claim it.",
        "Your payment failed. Please update your credit card information.",
        "Security alert! Confirm your account by clicking this link.",
        "You won a lottery. Provide your personal information to receive your prize.",

        "Hi John, our meeting is scheduled for tomorrow at 10 AM.",
        "Your Amazon order has been shipped and will arrive tomorrow.",
        "Please find attached the project report for today's meeting.",
        "Your electricity bill is available in your account.",
        "The college timetable has been updated. Please check the portal.",
        "Thank you for submitting your assignment.",
        "Your appointment is confirmed for Monday at 3 PM.",
        "The monthly company newsletter is now available.",
        "Your GitHub repository has been successfully updated.",
        "Please review the document and send your feedback."
    ],

    # 1 = Phishing
    # 0 = Safe
    "label": [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

print("\n==============================")
print(" PHISHING EMAIL DETECTOR")
print("==============================")

print("\nDataset:")
print(df.head())

# ==========================================
# 2. EXTRACT URL FEATURES
# ==========================================

def extract_features(text):

    # Number of URLs
    url_count = len(
        re.findall(r'https?://\S+|www\.\S+', text)
    )

    # Number of suspicious words
    suspicious_words = [
        "urgent",
        "verify",
        "password",
        "account",
        "winner",
        "prize",
        "click",
        "bank",
        "suspended",
        "credit card",
        "login",
        "security"
    ]

    suspicious_count = 0

    text_lower = text.lower()

    for word in suspicious_words:
        if word in text_lower:
            suspicious_count += 1

    return url_count, suspicious_count


df["url_count"] = df["email"].apply(
    lambda x: extract_features(x)[0]
)

df["suspicious_words"] = df["email"].apply(
    lambda x: extract_features(x)[1]
)

print("\nExtracted Features:")
print(df[["email", "url_count", "suspicious_words", "label"]])

# ==========================================
# 3. SPLIT DATA
# ==========================================

X = df["email"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. CONVERT TEXT INTO NUMBERS
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# ==========================================
# 5. TRAIN MACHINE LEARNING MODEL
# ==========================================

model = LogisticRegression()

model.fit(
    X_train_vectorized,
    y_train
)

print("\nModel training completed!")

# ==========================================
# 6. TEST MODEL
# ==========================================

y_prediction = model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    y_prediction
)

print("\n==============================")
print(" MODEL PERFORMANCE")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_prediction,
        target_names=["Safe", "Phishing"]
    )
)

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection - Confusion Matrix")

plt.show()

# ==========================================
# 8. PREDICT A NEW EMAIL
# ==========================================

print("\n==============================")
print(" TEST A NEW EMAIL")
print("==============================")

new_email = input(
    "\nEnter an email/message to check: "
)

new_email_vectorized = vectorizer.transform(
    [new_email]
)

prediction = model.predict(
    new_email_vectorized
)

probability = model.predict_proba(
    new_email_vectorized
)

if prediction[0] == 1:

    print("\n⚠️ RESULT: PHISHING EMAIL")

    print(
        f"Phishing Probability: "
        f"{probability[0][1] * 100:.2f}%"
    )

else:

    print("\n✅ RESULT: SAFE EMAIL")

    print(
        f"Safe Probability: "
        f"{probability[0][0] * 100:.2f}%"
    )

# ==========================================
# END
# ==========================================
