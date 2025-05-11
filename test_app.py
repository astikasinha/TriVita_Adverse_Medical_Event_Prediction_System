import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytest
from app import app, get_top_keywords, vectorizer, symptom_classifier_model

# Set up a test client
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test the home page loads
def test_home_page(client):
    response = client.get('/')
    print(response.data.decode())  # Add this line to inspect the response content
    assert response.status_code == 200
    assert b'Symptom Analysis Tool' in response.data
    assert b'Get Started' in response.data


# Test prediction functionality with valid input
def test_prediction_output():
    text = "I have fever and body ache"
    vector = vectorizer.transform([text])
    prediction = symptom_classifier_model.predict(vector)[0]
    assert prediction in symptom_classifier_model.classes_

# Test prediction functionality with empty input
def test_prediction_empty_input():
    text = ""
    vector = vectorizer.transform([text])
    prediction = symptom_classifier_model.predict(vector)[0]
    assert prediction in symptom_classifier_model.classes_

# Test prediction functionality with invalid input (e.g., non-symptom text)
def test_prediction_invalid_input():
    text = "I like reading books"
    vector = vectorizer.transform([text])
    prediction = symptom_classifier_model.predict(vector)[0]
    assert prediction in symptom_classifier_model.classes_

# Test get_top_keywords function
def test_top_keywords_function():
    text = "cough cold headache"
    keywords = get_top_keywords(text, symptom_classifier_model, vectorizer)
    assert isinstance(keywords, list)
    assert all(isinstance(k, tuple) for k in keywords)

# Test for empty input in get_top_keywords function
def test_top_keywords_empty_input():
    text = ""
    keywords = get_top_keywords(text, symptom_classifier_model, vectorizer)
    assert keywords == []

# Test if the vectorizer is working properly
def test_vectorizer():
    text = "fever"
    vector = vectorizer.transform([text])
    assert vector.shape[0] == 1  # Should have 1 sample
    assert vector.shape[1] > 0  # Should have non-zero number of features
