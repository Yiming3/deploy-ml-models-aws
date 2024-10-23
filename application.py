from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

# Load the models
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)
with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    prediction = loaded_model.predict(vectorizer.transform([text]))[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    application.run(port=5000, debug=True)
