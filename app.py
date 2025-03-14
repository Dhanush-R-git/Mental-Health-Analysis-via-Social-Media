from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import pandas as pd
import re
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from threading import Lock

# Load dataset and preprocess text
df = pd.read_csv("Data.csv")

def clean_text(text):
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters
    return text.lower().strip()

df['Cleaned_Comments'] = df['Comments'].apply(clean_text)

# Encode sentiment labels
label_encoder = LabelEncoder()
df['Encoded_Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Comments'], df['Encoded_Sentiment'], test_size=0.2, random_state=42)

# Train model (Logistic Regression with TF-IDF)
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Save trained model and label encoder
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Load trained model for Flask
model = joblib.load('emotion_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Flask app setup
app = Flask(__name__)
app.secret_key = '5000'  # Secret key for session handling

@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize emotion counts in session if they don't exist
    if 'emotion_counts' not in session:
        session['emotion_counts'] = {emotion: 0 for emotion in label_encoder.classes_}
    
    return render_template('index.html', counts=session['emotion_counts'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        data = json.load(file)  # Load JSON data
        
        # Reset emotion counts for this session
        session['emotion_counts'] = {emotion: 0 for emotion in label_encoder.classes_}
        session.modified = True  # Ensure session updates
        emotion_counts = session['emotion_counts']

        predictions = []
        
        # Extract comments and analyze sentiment
        for entry in data:
            if "string_map_data" in entry and "Comment" in entry["string_map_data"]:
                comment = entry["string_map_data"]["Comment"]["value"]
                cleaned_input = clean_text(comment)
                
                prediction = model.predict([cleaned_input])
                emotion = label_encoder.inverse_transform(prediction)[0]             
                emotion_counts[emotion] += 1
                
                predictions.append({'comment': comment, 'emotion': emotion})
        
        # Commit changes to session
        session['emotion_counts'] = emotion_counts
        session.modified = True  

        # Count positive and negative sentiments
        negative_count = emotion_counts.get('negative', 0)
        positive_count = emotion_counts.get('positive', 0)

        print(f"Negative Count: {negative_count}, Positive Count: {positive_count}")  # Debugging

        # Trigger questionnaire if negative count is 3 or more than positive count
        if negative_count >= positive_count + 3:
            return redirect(url_for('psychological_questions'))

        return jsonify({'predictions': predictions, 'counts': emotion_counts})

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON file'}), 400

@app.route('/psychological-questions', methods=['GET'])
def psychological_questions():
    questions = [
        {'question': 'Do you feel down or hopeless?', 'options': ['Yes', 'No', 'Maybe', 'Sometimes']},
        {'question': 'Do you have trouble finding joy in things you usually enjoy?', 'options': ['Yes', 'No', 'Maybe', 'Sometimes']},
        {'question': 'Do you often feel exhausted, even after rest?', 'options': ['Yes', 'No', 'Maybe', 'Sometimes']},
    ]
    return render_template('questions.html', questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
