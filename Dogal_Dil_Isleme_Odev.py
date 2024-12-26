# Gerekli Kütüphanelerin Eklenmesi

import pandas as pd
import synonyms
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import regex as re
from collections import Counter
import zeyrek
nltk.download('stopwords')
nltk.download('punkt_tab')

# Excel Bağlantısının Yapılması
input_file_path = 'Reviews.xlsx'
stop_words_file_path = 'stop_words.py' # İnternetten alınmış ekstra gereksiz kelimeler dosyası
df = pd.read_excel('Reviews.xlsx')

# Verilerin Küçük Harfe Çevrilmesi
df['Reviews'] = df['Reviews'].str.lower()

# Verilerdeki Gereksiz Noktalama İşaretleri ve Boşlukların Kaldırılması
df['Reviews'] = df['Reviews'].str.replace(r'[^\w\s]', '', regex=True)

# Gereksiz Kelimelerin Kaldırılması
sw = set(stopwords.words('turkish'))

with open(stop_words_file_path, encoding='utf-8') as f:
    custom_stopwords = set(f.read().splitlines())

all_stopwords = sw.union(custom_stopwords)
all_stopwords.add("bir")
all_stopwords.add("bu")
all_stopwords.add("Bu")

df['Reviews'] = df['Reviews'].apply(lambda x: " ".join(word for word in str(x).split() if word not in all_stopwords))

temp_df = pd.Series(' '.join(df['Reviews']).split()).value_counts() # Nadir Kelimeleri Belirleme
drops = temp_df[temp_df <= 1].index.tolist()                        # Nadir Kelimeleri Belirleme
df['Reviews'] = df['Reviews'].apply(lambda x: " ".join(word for word in str(x).split() if word not in drops)) # Nadir Kelimeleri Çıkarma

# Tokenization İşlemi
analyzer = zeyrek.MorphAnalyzer()

kelime_deseni = r'\b\w+\b'

df['Reviews_token'] = df['Reviews'].apply(lambda x: [word for word in re.findall(kelime_deseni, str(x)) if analyzer.analyze(word)])

# Lemmatization İşlemi
df['Reviews_lemma'] = df['Reviews'].apply(lambda x: [analyzer.lemmatize(word)[0][1][0] for word in x.split()])

# Lemmatizated Verileri Küçük Harfe Çevirme
df['Reviews_lemma_lower'] = df['Reviews_lemma'].apply(lambda x: [word.lower() for word in x])

# Lemmatizated Verilerden Kendini Tekrar Eden ve Eş Anlamlı Verilerin Silinmesi
df['Reviews_lemma_clean'] = df['Reviews_lemma_lower'].apply(lambda x: [re.sub(r'[^\w\s]|[0-9]', '', word) for word in x])

lemmatized_output = df['Reviews_lemma_clean'].apply(lambda text: text.split() if isinstance(text, str) else [])

word_list = [word for sublist in lemmatized_output for word in sublist if word not in all_stopwords]
counter = Counter(word_list)
lemmatized_output_normalized = [(synonyms.get(word, word), count) for word, count in counter.most_common()]

# Kelime Frekansı Hesaplanması
from collections import Counter
all_words = [word for sublist in df['Reviews_lemma_clean'] for word in sublist] # Tüm kelimeleri listeye koyar.
word_counts = Counter(all_words) # Kelime Frekansı Hesaplanır.
top_words = word_counts.most_common(10) # En Sık Kullanılan 10 kelimeyi alır.

# En Çok Kullanılan Kelimeler Bar Grafiği
words = [word for word, count in top_words]
counts = [count for word, count in top_words]
plt.bar(words, counts)
plt.xlabel('Kelimeler')
plt.ylabel('Frekans')
plt.title('En Çok Kullanılan Kelimeler')
plt.xticks(rotation=45, ha='right')
plt.show()

# En Çok Kullanılan Olumsuz Kelimeler
olumsuz_kelimeler = ["kötü", "berbat", "rezalet", "olumsuz", "şaşırttı", "beğenmedim", "karşı", "iade", "sorun", "hata", "hasarlı", "çalışmıyor", "eksik", "geç", "yavaş", "kalitesiz", "rahatsız", "gürültülü", "pis"]  # Olumsuz kelimeleri buraya ekleyin
olumsuz_kelime_sayilari = Counter()

for yorum in df['Reviews_lemma_clean']:
    for kelime in yorum:
        if kelime in olumsuz_kelimeler:
            olumsuz_kelime_sayilari[kelime] += 1

en_cok_kullanilan_olumsuz_kelimeler = olumsuz_kelime_sayilari.most_common(10)
kelimeler = [kelime for kelime, sayi in en_cok_kullanilan_olumsuz_kelimeler]
sayilar = [sayi for kelime, sayi in en_cok_kullanilan_olumsuz_kelimeler]
plt.bar(kelimeler, sayilar)
plt.xlabel('Olumsuz Kelimeler')
plt.ylabel('Frekans')
plt.title('En Çok Kullanılan Olumsuz Kelimeler')
plt.xticks(rotation=45, ha='right')
plt.show()

# Kelime Bulutu
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Yapılan İşlemler ile Yeni Excel Oluşturma
output_file_path = 'ReviewsFinal22030131115.xlsx'
df.to_excel(output_file_path, index=False)
print("Lemmatize edilmiş veriler başarıyla Excel dosyasına aktarıldı.")
df = pd.read_excel(input_file_path)










