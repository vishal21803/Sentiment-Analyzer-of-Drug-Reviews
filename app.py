from flask import Flask, render_template, request, redirect
import requests
import os

app = Flask(__name__)


API_URL = "https://huggingface.co/vish21803/drs-new-model"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

def analyze_sentiment(text: str):
    try:
        # Call HuggingFace API
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": text}, timeout=10)
        print("Status code:", response.status_code)
        print("Response text:", response.text)
        result = response.json()
        
        # DEBUG: print the raw API response
        print("API response:", result)

    except Exception as e:
        print(f"API error: {e}")
        return {"sentiment": "Neutral", "rating": 3, "confidence": 50.0}

    # Now continue processing the result...
    if isinstance(result, list) and len(result) > 0:
        label = result[0]["label"]
        score = float(result[0]["score"])
    else:
        label = "neutral"
        score = 0.5

    # Optional keyword-based neutral detection
    neutral_keywords = ['average', 'okay', 'not great', 'not bad', 'somewhat', 'might try', 'moderate', 'mediocre']
    if any(k in text.lower() for k in neutral_keywords):
        rating = 3
        sentiment = "Neutral"
        confidence = 50.0
    else:
        if score >= 0.95:
            rating = 5
        elif score >= 0.80:
            rating = 4
        elif 0.40 <= score <= 0.60:
            rating = 3
        elif score >= 0.20:
            rating = 2
        else:
            rating = 1

        sentiment = "Positive" if "POS" in label.upper() else ("Negative" if "NEG" in label.upper() else "Neutral")
        confidence = round(score * 100, 2)

    return {
        "sentiment": sentiment,
        "rating": rating,
        "confidence": confidence
    }



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    drug_name = request.form.get('drug_name', '')
    review = request.form.get('review', '')

    print(f"\nAnalyzing review: {review} for drug: {drug_name}")
    result = analyze_sentiment(review)

    return render_template(
        'result.html',
        review=review,
        drug_name=drug_name,
        sentiment=result['sentiment'],
        rating=result['rating'],
        confidence=result['confidence']
    )


@app.route('/displayData')
def display_data():
    return redirect("http://127.0.0.1/drs/templates/displaYData.php")


if __name__ == '__main__':
    app.run(debug=True)








