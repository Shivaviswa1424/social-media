****Depression Detection on Social Media - Project Report**

**1. Introduction****
This project focuses on detecting symptoms of depression in user-generated social media content using Natural Language Processing (NLP) and Machine Learning. The application is designed with an interactive Streamlit web interface that allows users to input text and receive real-time predictions along with visual and emotional feedback.

The project utilizes Python libraries such as NLTK for preprocessing, Scikit-learn for model building, and Plotly for visualization. Enhanced UI styling is applied using custom CSS and animated emoji-based reactions to make the application more engaging and user-friendly.

**2. Technologies Used**
Python – Base programming language

Streamlit – Front-end UI for interactive predictions

NLTK – Natural language preprocessing (stopword removal, tokenization)

Scikit-learn – Machine Learning (Naive Bayes classifier)

Plotly – Visualization (pie charts for prediction confidence)

Pandas – Data manipulation and loading

Joblib – Model saving/loading

HTML/CSS – Custom design and animation for styling

**3. Project Structure & Files**
File	Purpose
app.py	Main Streamlit app with UI, styling, emoji feedback, and result visualization
model.py	Handles data preprocessing, model training, and prediction
depression_dataset1.csv	Dataset of social media posts labeled as 'suicide' or 'non-suicide'
requirements.txt	List of required packages to run the project
utils/preprocess.py (optional)	For external preprocessing logic (if modularized)
README.md	Overview, setup instructions, and usage guide

**4. Workflow / How It Works**
Data Cleaning:
The social media posts are cleaned using NLTK to remove stopwords, punctuation, and special characters.

Feature Extraction:
Cleaned text is transformed into numerical features using TF-IDF Vectorizer.

Model Training:
A Naive Bayes classifier is trained on the cleaned data.

Prediction:
The app accepts user input through a Streamlit UI, processes it in real time, and predicts if the message reflects depressive behavior.

Feedback Display:

Shows a face emoji reaction (😀 or 😞) based on result

Displays a pie chart indicating confidence levels

**Conclusion**
This project provides a lightweight and effective solution for identifying possible signs of depression in social media content. It blends machine learning, NLP, and a user-friendly UI to raise awareness and assist in mental health screening.
