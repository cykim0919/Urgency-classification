import os
import json
import re
import pandas as pd

# ------------------------------------------------------------
# 규칙 기반 긴급도 라벨링 키워드
# ------------------------------------------------------------
EMERGENCY = [
    "누전", "합선", "스파크", "불", "연기", "타는냄새", "타는 냄새",
    "화재", "그을음", "전기타는냄새", "전기 타는 냄새",
    "가스", "가스냄새", "가스 냄새", "가스샘", "가스 샘",
    "보일러", "감전",
    "침수", "물새", "물 샘", "누수", "유수", "층간누수",
    "위험", "위험함", "붕괴", "떨어짐", "파손", "균열", "깨짐"
]

MODERATE = [
    "고장", "작동안함", "작동 안함",
    "수리", "교체", "문제", "이상",
    "전등", "전등나감", "전등 고장",
    "문의", "막힘", "장애"
]


# ------------------------------------------------------------
# 긴급도 스코어 계산
# ------------------------------------------------------------
def get_emergency_level(text):
    if not text:
        return 0

    t = text.replace(" ", "")

    # HIGH
    for w in EMERGENCY:
        if w in t:
            return 2

    # MID
    for w in MODERATE:
        if w in t:
            return 1

    return 0


# ------------------------------------------------------------
# JSON 폴더 하나 처리 → rows 반환
# ------------------------------------------------------------
def load_json_folder(json_folder, campus_name):
    rows = []

    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(json_folder, filename)

        with open(filepath, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        title = data.get("title", "")
        timestamp = data.get("timestamp", "")

        # 파일 구조 캠퍼스별로 다름
        if campus_name == "chuncheon":
            content = data.get("content", "")
        else:
            content = data.get("poster", "")  # 삼척/도계는 poster가 내용

        combined_text = f"{title} {content}"

        emergency = get_emergency_level(combined_text)

        rows.append({
            "campus": campus_name,
            "id": data.get("id", ""),
            "title": title,
            "content": content,
            "timestamp": timestamp,
            "emergency": emergency
        })

    return rows


# ------------------------------------------------------------
# 전체 통합 CSV 생성
# ------------------------------------------------------------
def create_merged_csv():
    # 폴더 경로
    base = r"C:\Users\ch901\PycharmProjects\인공지능project\crawler\data"

    folders = {
        "chuncheon": os.path.join(base, "chuncheon"),
        "samcheok": os.path.join(base, "samcheok"),
        "dogye": os.path.join(base, "dogye")
    }

    all_rows = []

    # 캠퍼스별 JSON 처리
    for campus, folder_path in folders.items():
        if not os.path.exists(folder_path):
            print(f"[WARN] 폴더 없음: {folder_path}")
            continue

        print(f"[INFO] {campus} 처리 중...")
        rows = load_json_folder(folder_path, campus)
        all_rows.extend(rows)

    # DataFrame 생성
    df = pd.DataFrame(all_rows)

    output_path = r"C:\Users\ch901\PycharmProjects\인공지능project\csv\merged_labeled.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n[완료] 통합 CSV 생성됨 → {output_path}")
    print(f"[INFO] 총 {len(df)}개 데이터")


# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    create_merged_csv()
