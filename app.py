from flask import Flask, render_template, request
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load the Word2Vec model
word2vec_model = Word2Vec.load("word2vec_model.model")  # Update with your Word2Vec model file

# Load the trained Random Forest classifier from the pickle file
with open('random_forest (1).pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Convert input text into a word vector using Word2Vec
def document_vector(doc):
    tokenized_text = word_tokenize(doc)
    filtered_text = [word.lower() for word in tokenized_text if word.lower() not in stop_words]
    word_vectors = [word2vec_model.wv[word] for word in filtered_text if word in word2vec_model.wv.index_to_key]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        # Return a zero vector if no words in the document are in the vocabulary
        return np.zeros(word2vec_model.vector_size)


@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        word_vector = document_vector(text)
        prediction = rf_classifier.predict([word_vector])[0]
        result = "Stressed" if prediction == 1 else "Not Stressed"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
