from flask import Flask, request, jsonify
import gensim
import numpy as np

app = Flask(__name__)

# fastText modelini yükleyin (Türkçe için önceden eğitilmiş modeli indirin)
# https://fasttext.cc/docs/en/crawl-vectors.html adresinden Türkçe modelini indirebilirsiniz
# Model dosyası 'cc.tr.300.bin' olarak varsayılmıştır
model_path = 'cc.tr.300.bin'  # Model dosyasının yolu

# fastText modelini doğru şekilde yükleyin
model = gensim.models.fasttext.load_facebook_model(model_path)
# Veya sadece kelime vektörlerini yüklemek istiyorsanız:
# model = gensim.models.KeyedVectors.load_fasttext_format(model_path)

# Gizli kelime
hidden_word = "bilişim"

# Gizli kelimenin embedding'ini alın
hidden_embedding = model.wv[hidden_word]

# Tüm kelimeler arasından gizli kelimeye en yakın ve en uzak benzerlik skorlarını bulalım
all_similarities = []
for word in model.wv.index_to_key:
    if word != hidden_word:
        similarity = model.wv.similarity(hidden_word, word)
        all_similarities.append(similarity)

max_similarity = max(all_similarities)  # En yakın kelimenin benzerliği
min_similarity = min(all_similarities)  # En uzak kelimenin benzerliği

print(f"En yakın benzerlik skoru: {max_similarity}")
print(f"En uzak benzerlik skoru: {min_similarity}")

# Benzerlik skorlarını 0.30 ile 0.64 arasında sınırlayalım
# Eğer max_similarity 0.64'ten büyükse 0.64'e, min_similarity 0.30'dan küçükse 0.30'a ayarlayalım
max_similarity = min(max_similarity, 0.64)
min_similarity = max(min_similarity, 0.30)

# Cosine similarity hesaplama fonksiyonu
def calculate_similarity(word1, word2):
    if word1 == word2:
        return 1.0  # Kelimeler aynıysa benzerlik 1.0 olsun
    elif word1 in model.wv and word2 in model.wv:
        similarity_score = model.wv.similarity(word1, word2)
        return similarity_score
    else:
        return None  # Eğer kelime modelde yoksa None döneriz

# Skorları 2 ile 5000 arasında map'leme fonksiyonu
def map_score(similarity_score, min_similarity, max_similarity):
    if similarity_score == 1.0:
        return 1  # Kelimenin kendisiyle eşleşiyorsa mesafe 0 olsun
    elif similarity_score >= max_similarity:
        return 2  # En yakın kelime için mesafe 2 olsun
    elif similarity_score <= min_similarity:
        return 5000  # En uzak kelime için mesafe 5000 olsun
    else:
        # Min ve max similarity'e göre 2 ile 5000 arasında yeniden ölçeklendir
        mapped_score = 2 + (5000 - 2) * ((max_similarity - similarity_score) / (max_similarity - min_similarity))
        return int(mapped_score)

# API endpoint'i
@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    guessed_word = data['word']

    # Gizli kelime ile tahmin edilen kelimenin benzerlik skorunu hesapla
    similarity_score = calculate_similarity(hidden_word, guessed_word)

    if similarity_score is None:
        return jsonify({'error': 'Kelime modelde bulunamadı'}), 400

    # Benzerlik skorunu 2 ile 5000 arasında map'le
    distance_score = map_score(similarity_score, min_similarity, max_similarity)

    return jsonify({'similarity': float(similarity_score), 'distance': distance_score})

if __name__ == '__main__':
    app.run(debug=True)
