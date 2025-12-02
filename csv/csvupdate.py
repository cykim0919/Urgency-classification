import pandas as pd

# === 1. 파일 경로 ===
csv1_path = "C:/Users/ch901/PycharmProjects/Urgency-classification/csv/final_result_kookje.csv"
csv2_path = "C:/Users/ch901/PycharmProjects/Urgency-classification/csv/merged_labeled.csv"

# === 2. CSV 읽기 ===
df1 = pd.read_csv(csv1_path, encoding="utf-8-sig")
df2 = pd.read_csv(csv2_path, encoding="utf-8-sig")

print("CSV1:", df1.shape)
print("CSV2:", df2.shape)

# === 3. 필요한 컬럼만 추려내기 ===
# title / content / emergency 3개로 통일
def normalize_columns(df):
    # 컬럼 이름 표준화
    df.columns = [c.lower().strip() for c in df.columns]

    # content가 없고 poster가 있는 경우 (삼척/도계 타입)
    if "content" not in df.columns and "poster" in df.columns:
        df["content"] = df["poster"]

    # text 생성
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = df["title"] + " " + df["content"]

    # emergency가 없는 경우 예외 처리
    if "emergency" not in df.columns:
        raise ValueError("CSV 파일에 emergency 라벨이 없습니다.")

    return df[["title", "content", "text", "emergency"]]


df1 = normalize_columns(df1)
df2 = normalize_columns(df2)

# === 4. 두 CSV 합치기 ===
df = pd.concat([df1, df2], axis=0, ignore_index=True)

print("\nMerged size:", df.shape)

# === 5. 전처리 ===
# 중복 제거
df = df.drop_duplicates(subset=["text"])
df = df.reset_index(drop=True)

# emergency 정수화
df["emergency"] = df["emergency"].astype(int)

# === 6. 최종 저장 ===
output_path = "final_merged_training.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n[완료] 학습용 CSV 생성됨 → {output_path}")
