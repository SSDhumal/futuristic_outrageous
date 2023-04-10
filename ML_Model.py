from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Load the preprocessed data
df=pd.read_csv("BankFAQs.csv")
df1=pd.read_csv("BankFAQs_1.csv")

data1=pd.concat([df1,df])

# Define the TD-IDF vectorizer and fit it to the data
tdidf = TfidfVectorizer()
tdidf.fit(data1['Question'].str.lower())

# Define the support vector machine model and fit it to the data
svc_model = SVC(kernel='linear')
svc_model.fit(tdidf.transform(data1['Question'].str.lower()), data1['Class'])

# Define a function to get the answer to a given question
def get_answer(question):
    # Vectorize the question
    question_tdidf = tdidf.transform([question.lower()])
    
    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(question_tdidf, tdidf.transform(data1['Question'].str.lower()))

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Get the predicted class of the query
    predicted_class = svc_model.predict(question_tdidf)[0]
    
    # If the predicted class is not the same as the actual class, return an error message
    if predicted_class != data1.iloc[most_similar_idx]['Class']:
        return {'error': 'Could not find an appropriate answer.'}
    
    # Get the answer and construct the response
    answer = data1.iloc[most_similar_idx]['Answer']
    response = {
        'answer': answer,
        'predicted_class': predicted_class
    }
    
    return response


app = Flask(__name__,template_folder='templates')

# Define route for chatbot web interface


@app.route('/')
def index():
    return render_template('layout.html')


# Define API route for answer prediction

@app.route('/predict', methods=['POST'])
def predict():
    question = request.form['question']

    response = get_answer(question)

    return jsonify(response)


if __name__ == '__main__':
     app.run(debug= True)
