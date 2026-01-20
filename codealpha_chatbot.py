import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load FAQ data
with open("faqs.json", "r") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# 2. Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

cleaned_questions = [clean_text(q) for q in questions]

# 3. Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(cleaned_questions)

# 4. Chat loop
print("🤖 FAQ Chatbot is running (type 'exit' to stop)")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye 👋")
        break

    cleaned_input = clean_text(user_input)
    user_vector = vectorizer.transform([cleaned_input])

    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarity.argmax()

    print("Bot:", answers[best_match_index])
