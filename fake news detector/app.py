from flask import Flask, render_template_string, request
import joblib
import re
import string

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 60px auto; }
        textarea { width: 100%; height: 150px; padding: 10px; }
        button { padding: 10px 20px; margin-top: 10px; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; border-radius: 8px; font-weight: bold; }
        .fake { background: #ffdddd; color: #a10000; }
        .real { background: #ddffdd; color: #007a00; }
    </style>
</head>
<body>
    <h2>📰 Fake News Detector</h2>
    <form method="POST">
        <textarea name="news_text" placeholder="Paste news article text here...">{{ text or '' }}</textarea><br>
        <button type="submit">Check News</button>
    </form>
    {% if result %}
    <div class="result {{ 'fake' if result == 'FAKE' else 'real' }}">
        Prediction: {{ result }} ({{ confidence }}% confidence)
    </div>
    {% endif %}
</body>
</html>
"""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    text = None
    if request.method == "POST":
        text = request.form["news_text"]
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]
        confidence = round(max(model.predict_proba(vec)[0]) * 100, 2)
    return render_template_string(HTML, result=result, confidence=confidence, text=text)

if __name__ == "__main__":
    app.run(debug=True)