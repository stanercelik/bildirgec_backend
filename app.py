import fasttext
import random
from flask import Flask, request, jsonify
import tdk.gts

app = Flask(__name__)

# FastText modelini yüklüyoruz
model_path = 'cc.tr.300.bin'
model = fasttext.load_model(model_path)

# Turkish words from words.txt
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

# Filtre: Sadece modelde olan ve words.txt'deki kelimeleri dikkate alacağız
valid_words_in_model = [word for word in turkish_words if model.get_word_vector(word) is not None]

# Randomly select a hidden word from the list, ensuring it's not a proper noun
def select_hidden_word(words):
    while True:
        word = random.choice(words)
        if not word[0].isupper() and model.get_word_vector(word) is not None:
            return word
        print(f"Skipping word not in model or proper noun: {word}")

# Set the hidden word
hidden_word = select_hidden_word(valid_words_in_model)
print(f"Selected hidden word: {hidden_word}")

# Tüm kelimeleri benzerliklerine göre sıralıyoruz
def rank_words_by_similarity(model, target_word, word_list):
    similarities = []
    target_vector = model.get_word_vector(target_word)

    for word in word_list:
        if word != target_word:  # Gizli kelimeyi dışarıda bırakıyoruz
            word_vector = model.get_word_vector(word)
            # Cosine similarity hesaplama
            similarity = cosine_similarity(target_vector, word_vector)
            similarities.append((word, similarity))

    # Benzerliklerine göre sıralıyoruz
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# Cosine similarity fonksiyonu
import numpy as np

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Calculate the rank of the guessed word
def calculate_rank(guessed_word, similarities):
    for idx, (word, _) in enumerate(similarities):
        if word == guessed_word:
            return idx + 2  # Sıra 1'den başlasın (0 değil)
    return None

# Tüm kelimeleri sıralıyoruz
ranked_similarities = rank_words_by_similarity(model, hidden_word, valid_words_in_model)

# İlk 10 kelimeyi yazdırıyoruz
print(f"The top 10 most similar words to '{hidden_word}' are:")
for word, similarity in ranked_similarities[:10]:
    print(f"{word}: {similarity:.4f}")

# Mastar kontrolü için yardımcı fonksiyon
def normalize_infinitive(word, word_list):
    suffixes = [('mek', 'me'), ('mak', 'ma')]
    for old_suffix, new_suffix in suffixes:
        if word.endswith(old_suffix):
            normalized = word[:-3] + new_suffix
            # Sadece normalize edilmiş hali kelime listesinde varsa değişikliği kabul et
            if normalized in word_list:
                return normalized
    return word

# API endpoint for similarity rank
@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    guessed_word = data['word']
    
    # Tahmin edilen kelimeyi ve gizli kelimeyi normalize et
    # valid_words_in_model listesini kontrol listesi olarak kullan
    normalized_guess = normalize_infinitive(guessed_word, valid_words_in_model)
    normalized_hidden = normalize_infinitive(hidden_word, valid_words_in_model)

    if normalized_guess == normalized_hidden:
        rank = 1
    else:
        # Normalize edilmiş kelimeyi ara
        rank = None
        for idx, (word, _) in enumerate(ranked_similarities):
            if normalize_infinitive(word, valid_words_in_model) == normalized_guess:
                rank = idx + 2
                break

    if rank is None:
        return jsonify({'error': 'Kelime modelde bulunamadı veya sıralanamadı'}), 400

    return jsonify({'rank': rank, 'total_words': len(ranked_similarities)})

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
    
@app.route('/hint', methods=['GET'])
def hint():
    try:
        hint_rank = int(request.args.get('rank'))
        
        if 0 < hint_rank <= len(ranked_similarities):
            hint_word = ranked_similarities[hint_rank - 1][0]
            return jsonify({'hint_word': hint_word, 'rank': hint_rank})
        else:
            return jsonify({'error': 'Geçersiz ipucu aralığı'}), 400
    except Exception as e:
        print(f"Error in hint: {e}")
        return jsonify({'error': 'Bir hata oluştu'}), 500
    
# API endpoint to return the closest 500 words
@app.route('/closest-words', methods=['GET'])
def get_closest_words():
    try:
        top_499_similar_words = ranked_similarities[:499]

        closest_words = [{'word': hidden_word, 'rank': 1}]

        closest_words += [
            {'word': word, 'rank': idx + 2}
            for idx, (word, _) in enumerate(top_499_similar_words)
        ]

        return jsonify({'closest_words': closest_words})
    except Exception as e:
        print(f"Error fetching closest words: {e}")
        return jsonify({'error': 'Error fetching closest words'}), 500

if __name__ == '__main__':
    app.run(debug=True)
