from flask import Flask, render_template, request, jsonify
import pandas as pd
from transformers import pipeline
from colorama import Fore, Style

app = Flask(__name__)

# Load IMDb dataset
df = pd.read_csv('C:/Users/saihi/OneDrive/Desktop/movie/IMDB Dataset.csv')

# Normalize column names (safe fix)
df.columns = df.columns.str.lower().str.strip()

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")


@app.route('/')
def home():
    print(Fore.GREEN + "[INFO] Home page accessed" + Style.RESET_ALL)
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # ✅ Read JSON from frontend (not form data)
        data = request.get_json()
        movie_name = data.get('movie_name') or data.get('movie') or ''
        description = data.get('description', '')

        if not description.strip():
            print(Fore.RED + "[ERROR] Description missing!" + Style.RESET_ALL)
            return jsonify({'error': 'Please enter your review!'}), 400

        # Run sentiment analysis
        sentiment = sentiment_pipeline(description)[0]
        sentiment_label = sentiment.get('label', 'NEUTRAL')
        score = sentiment.get('score', 0)

        # Convert model output to rating (1–5)
        if sentiment_label == "POSITIVE":
            predicted_rating = round(score * 5)
        else:
            predicted_rating = round((1 - score) * 5)

        predicted_rating = max(1, min(predicted_rating, 5))
        stars = '★' * predicted_rating + '☆' * (5 - predicted_rating)

        # ✅ Safe IMDb lookup (handles case and column differences)
        imdb_rating = "N/A"
        if 'movie name' in df.columns and 'movie rating' in df.columns:
            match = df.loc[
                df['movie name'].str.lower() == movie_name.lower(), 'movie rating'
            ]
            if not match.empty:
                imdb_rating = match.values[0]

        print(
            Fore.YELLOW
            + f"[ANALYSIS] Sentiment: {sentiment_label}, Score: {score:.2f}, Rating: {predicted_rating}/5"
            + Style.RESET_ALL
        )

        # ✅ Return JSON to frontend
        return jsonify({
            'sentiment': sentiment_label,
            'movie': movie_name,
            'predicted_rating': predicted_rating,
            'stars': stars,
            'imdb_rating': imdb_rating
        })

    except Exception as e:
        print(Fore.RED + f"[ERROR] Exception occurred: {e}" + Style.RESET_ALL)
        return jsonify({'error': 'Something went wrong on the server'}), 500


if __name__ == '__main__':
    print(Fore.YELLOW + "[INFO] Starting Flask server..." + Style.RESET_ALL)
    app.run(debug=True)
