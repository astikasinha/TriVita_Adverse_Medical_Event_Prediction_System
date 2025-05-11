# Adverse_Medical_Prediction_System

A machine learning–powered tool that classifies user-described symptoms into categories like mild, adverse, or critical using natural language processing. It highlights the top keywords contributing to the prediction and automatically generates a detailed PDF report for medical or analytical use.

 Project Objective
The primary goal of this project is to develop an automated symptom classification and reporting system. This helps medical professionals, researchers, or digital health platforms analyze symptom-related texts and classify their severity. Additionally, the system generates a comprehensive PDF report that includes:

The predicted symptom category.

The probability of the classification.

Keywords that contributed most to the prediction.

 Machine Learning Pipeline
Data Cleaning: Handled missing values from the symptom dataset.

Vectorization: Used TfidfVectorizer to transform symptom descriptions into numerical features.

Model: Trained a Logistic Regression model to classify symptoms.

Keyword Importance: Extracted top TF-IDF-weighted features contributing to prediction.

PDF Reporting: Generated a structured and printable PDF using the FPDF library.

 Dataset Overview
The training data consists of two columns:

text: Symptom description (e.g., "paralysis in limbs")

label: Category of severity (e.g., mild, adverse, critical)



Sample Input & Output
Input:

"Paralysis in limbs"

 Output:
Predicted Label: adverse

Top Contributing Keywords:

limbs: 0.3342

paralysis: 0.3342

in: 0.1019

Prediction Probabilities:

mild: 0.12

adverse: 0.76

critical: 0.12

PDF Report: symptom_report.pdf (includes all the above)

fpdf
pytest
