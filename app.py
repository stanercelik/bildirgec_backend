import gensim
import random
from flask import Flask, request, jsonify
import tdk.gts

app = Flask(__name__)

model_path = 'cc.tr.300.bin'
model = gensim.models.fasttext.load_facebook_model(model_path)

# Function to load Turkish words from words.txt 
def fetch_turkish_words_from_file():
    try:
        with open('words.txt', 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    except FileNotFoundError:
        print("Error: words.txt not found.")
        return []


turkish_words = fetch_turkish_words_from_file()

if not turkish_words:
    print("No Turkish words fetched from words.txt, using fallback list.")
    turkish_words = ["merhaba", "dünya", "kitap", "bilgisayar"]

# Randomly select a hidden word from the list, ensuring it's not a proper noun (doesn't start with an uppercase letter)
def select_hidden_word(words):
    while True:
        word = random.choice(words)
        if not word[0].isupper():
            return word
        print(f"Skipping proper noun: {word}")

# Set the hidden word
hidden_word = select_hidden_word(turkish_words)
print(f"Selected hidden word: {hidden_word}")

# Get the embedding of the hidden word
hidden_embedding = model.wv[hidden_word]

# Print the top 10 most similar words to the hidden word
most_similar_words = model.wv.most_similar(hidden_word, topn=10)
print(f"The top 10 most similar words to '{hidden_word}' are:")
for word, similarity in most_similar_words:
    print(f"{word}: {similarity:.4f}")

# Find the max and min similarities for mapping
all_similarities = [model.wv.similarity(hidden_word, word) for word in model.wv.index_to_key if word != hidden_word]
max_similarity = max(all_similarities)
min_similarity = min(all_similarities)

# Function to calculate cosine similarity
def calculate_similarity(word1, word2):
    if word1 == word2:
        return 1.0
    elif word1 in model.wv and word2 in model.wv:
        return model.wv.similarity(word1, word2)
    else:
        return None

def map_score(similarity_score, min_similarity, max_similarity):
    if similarity_score == 1.0:
        return 1
    else:
        mapped_score = 2 + (15000 - 2) * ((max_similarity - similarity_score) / (max_similarity - min_similarity))
        return max(2, min(15000, int(mapped_score)))

# API endpoint for similarity
@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    guessed_word = data['word']

    similarity_score = calculate_similarity(hidden_word, guessed_word)

    if similarity_score is None:
        return jsonify({'error': 'Kelime modelde bulunamadı'}), 400

    distance_score = map_score(similarity_score, min_similarity, max_similarity)

    return jsonify({'similarity': float(similarity_score), 'distance': distance_score})

# API endpoint to reveal the hidden word
@app.route('/reveal', methods=['GET'])
def reveal():
    return jsonify({'hidden_word': hidden_word})

# API endpoint to get the meaning of the word
@app.route('/meaning', methods=['GET'])
def get_word_meaning():
    word = request.args.get('word')
    if not word:
        return jsonify({'error': 'No word provided'}), 400
    try:
        results = tdk.gts.search(word)
        if results:
            meanings = [entry.meaning for entry in results[0].meanings]
            return jsonify({'meanings': meanings})
        else:
            return jsonify({'meanings': []}), 404
    except Exception as e:
        print(f"Error fetching meaning: {e}")
        return jsonify({'error': 'Error fetching meaning'}), 500

if __name__ == '__main__':
    app.run(debug=True)
