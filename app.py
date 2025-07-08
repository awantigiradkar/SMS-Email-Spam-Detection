import streamlit as st
import pickle
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PyPDF2 import PdfReader
import docx

ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Sidebar ---
st.sidebar.title("SMS/Email Spam Detector")

# 1. About
st.sidebar.markdown("### About")
st.sidebar.info("This app classifies SMS/Email messages as **Spam** or **Not Spam** using NLP and Machine Learning.")

# 2. Model Info
st.sidebar.markdown("### Model Info")
st.sidebar.code("Model: Multinomial Naive Bayes\nVectorizer: TF-IDF")

# 3. Feedback
st.sidebar.markdown("### Feedback")
feedback = st.sidebar.radio("Was the prediction correct?", ["👍 Yes", "👎 No"])

# 6. File Upload for Batch Prediction
st.sidebar.markdown("### Upload Message File")
uploaded_file = st.sidebar.file_uploader("Upload .txt / .pdf / .docx", type=["txt", "pdf", "docx"])

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8").splitlines()
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return [page.extract_text() for page in reader.pages if page.extract_text()]
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return [para.text for para in doc.paragraphs if para.text.strip()]
    else:
        return []

if uploaded_file is not None:
    messages = extract_text(uploaded_file)
    if messages:
        st.subheader("File Message Predictions")
        transformed_texts = [transform_text(msg) for msg in messages]
        vectors = tfidf.transform(transformed_texts)
        predictions = model.predict(vectors)
        result_df = pd.DataFrame({
            "Message": messages,
            "Prediction": ["Spam" if p == 1 else "Not Spam" for p in predictions]
        })
        st.dataframe(result_df)
    else:
        st.warning("No readable text found in the uploaded file.")

# --- Main Interface ---
st.title("Email/SMS Spam Classifier")

col1, col2 = st.columns([3, 1])
with col1:
    input_sms = st.text_area("Enter the message below:")
with col2:
    st.write("")
    st.write("")
    predict_button = st.button("Predict")

if predict_button and input_sms.strip() != "":
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.markdown("**Spam Detected!**")
        st.success("This message is likely **Spam**.")
        st.markdown("<p style='color:red; font-size: 18px;'>Be cautious before clicking any links or replying.</p>", unsafe_allow_html=True)
    else:
        st.markdown("**Safe Message**")
        st.info("This message is **Not Spam**.")
        st.markdown("<p style='color:green; font-size: 18px;'>Looks good. You can trust this message.</p>", unsafe_allow_html=True)
