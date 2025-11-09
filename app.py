from flask import Flask, render_template, request, redirect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

app = Flask(__name__)

# Model path
MODEL_PATH = r"vish21803/drs-new-model"

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model safely
print("Loading model...")
try:
    # Load config with trust_remote_code for custom models
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, num_labels=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Load model safely
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e


def get_rating_and_sentiment(probability: float):
    """Convert probability to rating and sentiment"""
    if probability >= 0.95:
        rating = 5
    elif probability >= 0.80:
        rating = 4
    elif 0.40 <= probability <= 0.60:
        rating = 3
    elif probability >= 0.20:
        rating = 2
    else:
        rating = 1

    if probability >= 0.80:
        sentiment = "Positive"
    elif probability <= 0.40:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return rating, sentiment


def analyze_sentiment(text: str):
    """Analyze sentiment of a given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        positive_prob = float(probabilities[0][1])

    # Detect neutral phrases
    neutral_keywords = [
        'average', 'okay', 'not great', 'not bad', 'somewhat', 
        'might try', 'moderate', 'mediocre'
    ]
    
    if any(keyword in text.lower() for keyword in neutral_keywords):
        rating = 3
        sentiment = "Neutral"
        confidence = 50.0
    else:
        rating, sentiment = get_rating_and_sentiment(positive_prob)
        confidence = round(positive_prob * 100, 2)

    return {
        'sentiment': sentiment,
        'rating': rating,
        'confidence': confidence,
        'raw_probability': positive_prob
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
