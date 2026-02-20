import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- FAQ DATA ----------------
faqs = {

    # AI Basics
    "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "why is ai important": "AI helps automate tasks, improve decision-making, and solve complex problems.",
    "what are examples of ai": "Examples of AI include chatbots, recommendation systems, self-driving cars, and virtual assistants.",
    "what is generative ai": "Generative AI creates new content such as text, images, music, or code.",
    "what is narrow ai": "Narrow AI is designed to perform a specific task, like facial recognition.",
    "what is general ai": "General AI refers to machines that can perform any intellectual task that humans can do.",

    # Machine Learning
    "what is machine learning": "Machine Learning is a subset of AI that enables systems to learn from data.",
    "types of machine learning": "The main types are supervised learning, unsupervised learning, and reinforcement learning.",
    "what is supervised learning": "Supervised learning uses labeled data to train models.",
    "what is unsupervised learning": "Unsupervised learning finds patterns in unlabeled data.",
    "what is reinforcement learning": "Reinforcement learning trains models using rewards and penalties.",
    "difference between ai and ml": "AI is the broader concept of intelligent machines, while ML is a subset that focuses on learning from data.",

    # Deep Learning
    "what is deep learning": "Deep Learning is a subset of Machine Learning that uses neural networks.",
    "what is a neural network": "A neural network is a model inspired by the human brain to recognize patterns.",
    "what is cnn": "CNN stands for Convolutional Neural Network, commonly used for image processing.",
    "what is rnn": "RNN stands for Recurrent Neural Network, used for sequential data like text.",

    # NLP
    "what is nlp": "Natural Language Processing (NLP) enables computers to understand human language.",
    "applications of nlp": "NLP is used in chatbots, translation systems, sentiment analysis, and speech recognition.",
    "what is tokenization": "Tokenization is the process of breaking text into smaller units like words.",
    "what is sentiment analysis": "Sentiment analysis determines whether text expresses positive, negative, or neutral sentiment.",

    # Data Science
    "what is data science": "Data Science involves extracting insights from data using statistics and machine learning.",
    "what is big data": "Big Data refers to extremely large datasets that require advanced tools to process.",
    "what is data mining": "Data mining is the process of discovering patterns in large datasets.",

    # Programming
    "what is python": "Python is a popular programming language used in web development, AI, and data science.",
    "why is python popular": "Python is popular because it is easy to learn, readable, and has many libraries.",
    "what is streamlit": "Streamlit is a Python library used to build web applications easily.",
    "what is scikit learn": "Scikit-learn is a Python library used for machine learning tasks."
}

# ---------------- PREPROCESSING ----------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

questions = [preprocess(q) for q in faqs.keys()]
answers = list(faqs.values())

# ---------------- VECTORIZE ----------------
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

# ---------------- CHATBOT FUNCTION ----------------
def chatbot(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    
    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match = similarity.argmax()
    best_score = similarity[0][best_match]

    if best_score < 0.3:
        return "Sorry, I don't have an answer for that question."
    
    return answers[best_match]

# ---------------- STREAMLIT UI ----------------
st.title("🤖 FAQ Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if st.button("Send"):
    if user_input.strip() != "":
        response = chatbot(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.write(f"🧑 **You:** {message}")
    else:
        st.write(f"🤖 **Bot:** {message}")