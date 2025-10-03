# Email Spam Classification with Logistic Regression

This Python console application classifies emails as **spam** or **legitimate** using a logistic regression model trained on a dataset of email features.

---

## Dataset

The dataset used is [`g_sharabidze2024_938274.csv`](https://github.com/your-username/your-repo-name/blob/main/g_sharabidze2024_938274.csv), which contains the following features:

- `words`: total word count in the email
- `links`: number of links (e.g., "http", "www")
- `capital_words`: number of fully capitalized words
- `spam_word_count`: number of spammy keywords (e.g., "free", "win", "money")
- `is_spam`: target label (1 = spam, 0 = legitimate)

---

## Model Training

### Step 1: Load and preprocess the data
```python
data = pd.read_csv("g_sharabidze2024_938274.csv")
X = data[["words", "links", "capital_words", "spam_word_count"]]
y = data["is_spam"]
```

### Step 2: Split into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 3: Train logistic regression model

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### Step 4: Print model coefficients

```python
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
```
---
## Model Evaluation

### Step 5: Predict and evaluate

```python
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```

## Confusion Matrix

[[TN FP]

[FN TP]]


## Accuracy

Accuracy: 94.13%

---
## Email Text Classification

### Step 6: Feature extraction from raw email

```python
def extract_features(email_text):
    words = len(email_text.split())
    links = email_text.lower().count("http") + email_text.lower().count("www")
    capital_words = sum(1 for word in email_text.split() if word.isupper())
    spammy_words = ["free", "win", "money", "offer", "urgent", "click"]
    spam_word_count = sum(email_text.lower().count(word) for word in spammy_words)
    return pd.DataFrame([[words, links, capital_words, spam_word_count]], columns=X.columns)

```

### Step 7: Classify new email

```python
def classify_email(email_text):
    features = extract_features(email_text)
    prediction = model.predict(features)[0]
    label = "Spam" if prediction == 1 else "Legitimate"
    print(f"Email Classification: {label}")

```

---
## Sample Emails

### Spam Email

`URGENT! Click here to WIN free money now! Visit http://spam.com for your offer.`

- Contains spammy keywords and a link
- Classified as Spam

### Legitimate Email

`Dear team, please find attached the report for Q3. Let me know if you have any questions.`

- Professional tone, no spammy words or links
- Classified as Legitimate

---

## Visualizations

### Spam vs Legitimate Distribution

```python
sns.countplot(x="is_spam", data=data)
```
- Shows class balance in the dataset

### Feature Correlation Heatmap

```python
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
```
- Reveals relationships between features and target label

## Requirements

pandas 2.1.1

numpy 1.26.0

scikit-learn 1.3.1

matplotlib 3.8.0

seaborn 0.13.0
