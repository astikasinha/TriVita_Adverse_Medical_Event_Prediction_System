from flask import Flask,request,jsonify,render_template,redirect,url_for,session,send_from_directory
import os
import numpy as np
from fpdf import FPDF
import uuid
from faster_whisper import WhisperModel
import joblib

whispermodel = WhisperModel("base",compute_type="int8")
# predictCustomModel=joblib.load('symptom_classifier_model_second.pkl')

model = joblib.load("symptom_classifier_model_second.pkl")
vectorizer = joblib.load("symptom_vectorizer.pkl")


def get_top_keywords(text, model, vectorizer, top_n=5):
    text_tfidf = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    class_index = model.predict_proba(text_tfidf).argmax()
    coefficients = model.coef_[class_index]
    word_indices = text_tfidf.toarray()[0].nonzero()[0]
    word_scores = [(feature_names[i], text_tfidf[0, i] * coefficients[i]) for i in word_indices]
    top_keywords = sorted(word_scores, key=lambda x: -abs(x[1]))[:top_n]
    return top_keywords


app = Flask(__name__)
app.secret_key='@abbc123*88$$$%251**vdka'
UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs("reports", exist_ok=True)
@app.route('/',methods=['GET',"POST"])
def home():
    transcription = session.pop('transcription', None)
    # analysedoutput = session.pop('analysedoutput', None)
    # predicted_label=session.pop('predicted_label',None),
    # keywords=session.pop('keywords',None),
    # probs=session.pop('probs',None),
    # return render_template('index.html',transcription=transcription,keywords=keywords,predicted_label=predicted_label,probs=probs)
    return render_template('index.html',transcription=transcription)


@app.route('/upload',methods=['POST'])
def uploadfile():
    if request.method=='POST':
        if 'file'not in request.files:
         return jsonify({'error': 'No file part in the request'}), 400 
        # data=request.get_json()
        file=request.files['file']
        if file.filename=='':
            return jsonify({'error':'No file selected'}),400
    
        filepath=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(filepath)
        print(filepath)
        segments,info=whispermodel.transcribe(filepath)
        transcriptedText=''.join([segment.text for segment in segments])
        session['transcription']=transcriptedText
    return redirect(url_for("home"))

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form.get('text')
    if not text:
        session['analysedoutput']='No text is input'
    else:
        prediction=predictCustomModel.predict([text])[0]
        session['analysedoutput']=prediction
    return redirect(url_for('home'))
# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import os
# import urllib.parse
# from faster_whisper import WhisperModel

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# model = WhisperModel("base", compute_type="int8")

# @app.route('/', methods=['GET'])
# def home():
#     transcription = request.args.get('transcription')
#     return render_template('index.html', transcription=transcription)

# @app.route('/upload', methods=['POST'])
# def uploadfile():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400


#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     segments, info = model.transcribe(filepath)
#     transcripted_text = ''.join([segment.text for segment in segments])

#     # Encode to safely include in URL
#     encoded = urllib.parse.quote_plus(transcripted_text)
#     return redirect(url_for('home', transcription=encoded))


@app.route('/analyze', methods=['POST'])
def analyze():
    print("analyze is called")
    text = request.form.get('text')
    probs_array = model.predict_proba(vectorizer.transform([text]))[0]
    class_labels = model.classes_
    probs = dict(zip(class_labels, probs_array))
    predicted_label = class_labels[np.argmax(probs_array)]
    keywords = get_top_keywords(text, model, vectorizer)
    filename = f"{uuid.uuid4().hex}.pdf"
    generate_report(text, predicted_label, keywords, probs, filename)
    session['predicted_label']=predicted_label
    keywordscore=keywords
    print("Debug Value: ",keywords)
    keywordsonly=[str(word) for word,score in keywordscore]
    print("keywordslist: ",keywordsonly)
    # session['keywords']=', '.join(keywordsonly)
    # session['probs']=probs[predicted_label]
    return jsonify({
        "predicted_label": predicted_label,
        "keywords": keywords,
        "probs": probs,
        "filename": filename
    })
    # return redirect(url_for('home'))



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
        pdf.cell(0, 10, f"  {label}: {prob*100:.2f}%", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Top Contributing Keywords:", ln=True)
    for word, score in keywords:
        pdf.cell(0, 10, f"  {word}: {score:+.4f}", ln=True)
    pdf.output(f"reports/{filename}")


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory("reports", filename, as_attachment=True)



if __name__=='__main__':
    app.run(debug=True)
