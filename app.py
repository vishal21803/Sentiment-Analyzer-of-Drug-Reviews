from flask import Flask, render_template, request, redirect
import requests

app = Flask(__name__)

# Your HuggingFace Space API endpoint
API_URL = "https://vish21803-drs-api.hf.space/run/predict"


def analyze_sentiment(text):
    """
    Sends text to your HuggingFace Space and returns:
    - sentiment (Positive/Negative/Neutral)
    - rating (1 to 5)
    - confidence (%)
    """

    try:
        response = requests.post(API_URL, json={"data": [text]}, timeout=20)
        result = response.json()

        # Extract expected data
        data = result["data"][0]

        return {
            "sentiment": data["sentiment"],
            "rating": data["rating"],
            "confidence": data["confidence"]
        }

    except Exception as e:
        print("API ERROR:", e)
        # fallback default
        return {
            "sentiment": "Neutral",
            "rating": 3,
            "confidence": 50.0
        }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    drug_name = request.form.get('drug_name', '')
    review = request.form.get('review', '')

    print(f"\nAnalyzing: {review}")

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
