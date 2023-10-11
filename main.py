from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data
X_train = [
    "I am very happy",
    "I feel great",
    "This is wonderful",
    "I am sad",
    "This is depressing",
    "I love this product",
    "This is terrible",
    "I'm thrilled with the results",
    "It's fantastic",
    "I can't stand this",
    "I'm so excited",
    "It's amazing",
    "I'm disappointed",
    "It's awesome",
    "This is superb",
    "I hate it",
    "I'm overjoyed",
    "It's a disaster",
    "I'm ecstatic",
    "This is the best",

]

y_train = ["positive", "positive", "positive", "negative", "negative", "positive", "negative", "positive", "positive", "negative", "positive", "positive", "negative", "positive", "positive", "negative", "positive", "negative", "positive", "positive"]


vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)

classifier = MultinomialNB()

# Train the model
classifier.fit(X_train, y_train)

# Input
new_review = ["Im mad and sad "]


new_review = vectorizer.transform(new_review)

predicted_category = classifier.predict(new_review)

print("Category:", predicted_category[0])