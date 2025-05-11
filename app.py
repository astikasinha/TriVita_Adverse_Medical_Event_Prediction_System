from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from fpdf import FPDF
import os
import joblib
import uuid
from faster_whisper import WhisperModel
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '@abbc123*88$$$%251**vdka'

# Setup absolute paths
BASE_DIR = r"C:\Users\astik\OneDrive\Desktop\VEERSANEW\veersa"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load models
whisper_model = WhisperModel("base", compute_type="int8")
symptom_classifier_model = joblib.load(os.path.join(BASE_DIR, "symptom_classifier_model_second.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "symptom_vectorizer.pkl"))

# Generate PDF report
def generate_report(text, predicted_label, keywords, probs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Symptom Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Input Text:\n{text}")
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 10, f"Predicted Label: {predicted_label}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(0, 10, "Class Probabilities:", ln=True)
    for label, prob in probs.items():
        pdf.cell(0, 10, f"  {label}: {prob * 100:.2f}%", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Top Contributing Keywords:", ln=True)
    for word, score in keywords:
        pdf.cell(0, 10, f"  {word}: {score:+.4f}", ln=True)

    # Save PDF
    pdf.output(os.path.join(REPORT_FOLDER, filename))

# Get top keywords
def get_top_keywords(text, model, vectorizer, top_n=5):
    text_tfidf = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    class_index = model.predict_proba(text_tfidf).argmax()
    coefficients = model.coef_[class_index]
    word_indices = text_tfidf.toarray()[0].nonzero()[0]
    word_scores = [(feature_names[i], text_tfidf[0, i] * coefficients[i]) for i in word_indices]
    return sorted(word_scores, key=lambda x: -abs(x[1]))[:top_n]

# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    transcription = session.pop('transcription', None)
    analysedoutput = session.pop('analysedoutput', None)
    report_filename = session.pop('report_filename', None)
    return render_template('index.html', transcription=transcription, analysedoutput=analysedoutput, report_filename=report_filename)

# Upload route
@app.route('/upload', methods=['POST'])
def uploadfile():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    segments, info = whisper_model.transcribe(filepath)
    transcripted_text = ''.join([segment.text for segment in segments])
    session['transcription'] = transcripted_text

    return redirect(url_for("home"))

# Predict symptoms
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    if not text:
        session['analysedoutput'] = 'No text input'
    else:
        text = str(text)
        text_vectorized = vectorizer.transform([text])
        prediction = symptom_classifier_model.predict(text_vectorized)[0]

        #prediction = symptom_classifier_model.predict([text])[0]
        session['analysedoutput'] = prediction

        probs_array = symptom_classifier_model.predict_proba(vectorizer.transform([text]))[0]
        class_labels = symptom_classifier_model.classes_
        probs = dict(zip(class_labels, probs_array))
        predicted_label = class_labels[np.argmax(probs_array)]
        keywords = get_top_keywords(text, symptom_classifier_model, vectorizer)

        filename = f"{uuid.uuid4().hex}.pdf"
        generate_report(text, predicted_label, keywords, probs, filename)
        session['report_filename'] = filename

    return redirect(url_for('home'))

# Download route
@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(REPORT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

# Start the app
if __name__ == '__main__':
    app.run(debug=True)
