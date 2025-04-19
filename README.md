# Disaster Management Tweet Classifier


This project, developed as part of an internship with Infosys, is an AI-based model for classifying Twitter tweets as real or fake disaster-related content. The primary objective is to provide emergency services and organizations with timely, relevant information for disaster response and awareness.

# Project Overview
During this internship, we focused on building a machine learning model that processes tweet data and determines the credibility of the content. The classifier analyzes tweets to distinguish genuine disaster reports from unrelated or misleading information, allowing for better resource allocation and response time during emergencies.

# Features
Data Collection and Preprocessing:

We gathered data from Twitter's API using keywords related to disasters.
Preprocessing steps included cleaning text, removing irrelevant characters, and handling emojis and links.

Tweet Classification Model:
A Natural Language Processing (NLP) model was created using machine learning algorithms to classify tweets.
The model analyzes language patterns commonly associated with disaster-related content.

Real-Time Analysis:
The model can analyze tweets in real-time, enabling continuous monitoring for real disaster incidents.

# Technologies Used
Python: For data processing, model training, and deployment.
Machine Learning Libraries: Used scikit-learn for model building, alongside pandas and NumPy for data manipulation.
Natural Language Processing: Implemented NLP techniques such as tokenization, TF-IDF vectorization, and stop word removal.
Twitter API: For collecting real-time tweet data on disaster events.


# Project Workflow
Data Collection: Collected and labeled tweets into "real" or "fake" categories based on relevance to disasters.
Data Preprocessing: Cleaned and prepared the data to ensure the model can process it effectively.
Model Training: Experimented with multiple machine learning models to find the most accurate classifier.
Testing & Evaluation: Evaluated the model using metrics like accuracy, precision, and recall.
Deployment: Implemented a basic framework for real-time tweet analysis using the trained model.


# Installation
Clone the repository:
    git clone :  (https://github.com/Vicky16032205/Infosys-internship.git)
    
Install required dependencies:
    pip install -r requirements.txt
    
Configure Twitter API keys (for real-time tweet collection).

# Usage

Training the Model:
Use the dataset of disaster tweets to train the model.
Execute the script train_model.py to start training.
Real-Time Classification:
Run tweet_classifier.py to collect and classify tweets in real time.


# Results
The final model achieved an accuracy of around 97% on the test set.
Precision and recall scores show that the classifier is effective at distinguishing real disaster tweets from unrelated ones.


# Future Work
Extended Dataset: Continuously update the dataset for more diverse disaster scenarios.
Improved Model: Explore advanced models like transformers or BERT for improved text classification.
Deployment as a Web Application: Build a front-end for the model to make it accessible for wider use.


# Acknowledgments
Special thanks to the Infosys Springboard team for their support and guidance throughout the project.
