import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

CSV_PATH = r"C:\Users\ch901\PycharmProjects\인공지능project\csv\merged_labeled.csv"
OUTPUT_KEYWORDS = r"C:\Users\ch901\PycharmProjects\인공지능project\keywords_emergency.txt"

okt = Okt()

# --------------------------
# 1. 형태소 분석 기반 토큰화 함수
# --------------------------
def tokenize(text):
    tokens = []
    for word, pos in okt.pos(text, stem=True):
        if pos in ["Noun", "Verb", "Adjective"]:  # 의미 있는 토큰만
            if len(word) > 1:  # 한 글자 토큰 제거 (불/불편 구분 문제 해결)
                tokens.append(word)
    return tokens

# --------------------------
# 2. 데이터 로딩
# --------------------------
df = pd.read_csv(CSV_PATH)

texts = df["content"].fillna("").tolist()
labels = df["emergency"].tolist()

# --------------------------
# 3. TF-IDF 벡터화
# --------------------------
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    max_features=5000
)

X = vectorizer.fit_transform(texts)
y = labels

# --------------------------
# 4. Logistic Regression 학습
# --------------------------
clf = LogisticRegression(max_iter=300)
clf.fit(X, y)

# --------------------------
# 5. 긴급도=2에서 가장 영향력 높은 단어 TOP 40 자동 추출
# --------------------------
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[2]  # 클래스 2에 대한 weight

top_n = 40
top_indices = np.argsort(coefs)[-top_n:]
top_words = [feature_names[i] for i in top_indices]

print("===== 긴급도(2) 대표 키워드 자동 추출 =====")
for w in reversed(top_words):
    print(w)

with open(OUTPUT_KEYWORDS, "w", encoding="utf-8") as f:
    for w in reversed(top_words):
        f.write(w + "\n")

print(f"\n키워드가 저장되었습니다: {OUTPUT_KEYWORDS}")
