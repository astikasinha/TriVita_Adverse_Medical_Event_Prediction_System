#  Adverse Medical Prediction System


#### Our Project is live at https://trivita-adverse-medical-event-prediction-1o7k.onrender.com

#### Working Prototype Video Link: https://vimeo.com/1083255406/80bbb9d551?share=copy

#### Presentation Link: https://vimeo.com/1083284085/77a28fd180?share=copy

#### Figma Link : https://www.figma.com/design/76YXjaHT7Kj3E9lETOmWJc/Adverse-Health-Event-Prediction-System?node-id=0-1&t=hk8aacMqHrrXKZ9b-1

#### Presentation PowerPoint Access: https://www.canva.com/design/DAGnINXrG-Y/G7b4IqCt5bVwm6eaqiTiHw/view?utm_content=DAGnINXrG-Y&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h9871d16a7e

A machine learning–powered NLP tool that classifies user-described symptoms into categories like **not serious**, **adverse**, or **moderate**. It highlights the most influential keywords contributing to the prediction and generates a **comprehensive PDF report** for medical or analytical use.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5af6133b-0b8d-4c29-9034-4a387f31bc94" />

---

##  Features

- Symptom severity classification (`mild`, `adverse`, `critical`)
- Keyword importance analysis (via TF-IDF)
- Probability distribution of predictions
- Auto-generated PDF report using `fpdf`


---

##  Objective

To provide healthcare professionals and researchers with an **automated pipeline** for:

- Classifying unstructured symptom descriptions.
- Highlighting contributing symptom keywords.
- Generating structured reports for offline or clinical use.

---



##  Machine Learning Pipeline

1. **Data Cleaning**  
   - Missing values handled.
   - Text preprocessing (lowercasing, punctuation removal, etc.).

2. **Text Vectorization**  
   - Using `TfidfVectorizer` from `scikit-learn`.

3. **Model Training**  
   - Trained a `Logistic Regression` classifier on labeled symptom data.

4. **Keyword Importance**  
   - Extracted top weighted TF-IDF terms contributing to classification.

5. **PDF Reporting**  
   - Used `fpdf` to generate clean, printable reports.

---

## Dataset Overview

| Column | Description                     |
|--------|---------------------------------|
| text   | Symptom description (e.g. "paralysis in limbs") |
| label  | Severity category (`not serious`, `adverse`, `moderate`) |

---

## Sample Input & Output

**Input:**
Paralysis in limbs
**Output:**
```yaml
Predicted Label: adverse

Top Contributing Keywords:
- limbs: 0.3342
- paralysis: 0.3342
- in: 0.1019

Prediction Probabilities:
- mild: 0.12
- adverse: 0.76
- critical: 0.12

PDF Report: Generated as symptom_report.pdf
