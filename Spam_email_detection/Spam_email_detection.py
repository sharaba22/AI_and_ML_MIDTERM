# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load the dataset
# The CSV file contains email features and a binary label indicating spam (1) or legitimate (0)
data = pd.read_csv("g_sharabidze2024_938274.csv")

# Select relevant features and target variable
# Features: number of words, links, capitalized words, and spammy words
# Target: is_spam (1 = spam, 0 = legitimate)
X = data[["words", "links", "capital_words", "spam_word_count"]]
y = data["is_spam"]

# Step 2: Split the data into training and testing sets
# 70% for training, 30% for testing to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a logistic regression model
# Logistic regression is suitable for binary classification tasks like spam detection
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Print model coefficients
# These values show how each feature influences the prediction
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# Step 5: Evaluate the model on the test set
# Predict labels for the test data
y_pred = model.predict(X_test)

# Calculate confusion matrix and accuracy
# Confusion matrix shows true/false positives and negatives
# Accuracy shows overall correct predictions
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nAccuracy: {accuracy:.2%}")

# Step 6: Define a function to extract features from raw email text
# This simulates real-world email parsing and feature engineering
def extract_features(email_text):
    words = len(email_text.split())
    links = email_text.lower().count("http") + email_text.lower().count("www")
    capital_words = sum(1 for word in email_text.split() if word.isupper())
    spammy_words = ["free", "win", "money", "offer", "urgent", "click"]
    spam_word_count = sum(email_text.lower().count(word) for word in spammy_words)
    return pd.DataFrame([[words, links, capital_words, spam_word_count]], columns=X.columns)

# Step 7: Classify a new email using the trained model
def classify_email(email_text):
    features = extract_features(email_text)
    prediction = model.predict(features)[0]
    label = "Spam" if prediction == 1 else "Legitimate"
    print(f"\nEmail Classification: {label}")
    print("Extracted Features:")
    print(features.to_string(index=False))

# Test with a manually created spam email
spam_email = "URGENT! Click here to WIN free money now! Visit http://spam.com for your offer."
classify_email(spam_email)

# Test with a manually created legitimate email
legit_email = "Dear team, please find attached the report for Q3. Let me know if you have any questions."
classify_email(legit_email)

# Visualization 1 - Spam vs Legitimate distribution
# Helps understand class balance in the dataset
plt.figure(figsize=(6, 4))
sns.countplot(x="is_spam", data=data)
plt.title("Spam vs Legitimate Email Distribution")
plt.xlabel("Email Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("spam_legit_distribution.png")
plt.show()

# Visualization 2 - Feature correlation heatmap
# Shows how features relate to each other and to the target label
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png")
plt.show()
