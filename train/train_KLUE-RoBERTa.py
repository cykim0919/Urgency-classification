import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# =========================
# 1. 설정값(경로/모델명 등)
# =========================
CSV_PATH = r"C:\Users\ch901\PycharmProjects\Urgency-classification\csv\final_merged_training.csv"
MODEL_NAME = "klue/roberta-base"# KLUE-roberta 사용
NUM_LABELS = 3                  # 긴급도 0,1,2
OUTPUT_DIR = r"/csv/roberta_emergency_model"


# =========================
# 2. seed 고정
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================
# 3. Dataset 정의
# =========================
class EmergencyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# =========================
# 4. 데이터 로딩 & 전처리
# =========================
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = df["title"] + " " + df["content"]

    df = df.dropna(subset=["text", "emergency"])
    df = df.reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["emergency"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["emergency"]
    )

    print("전체 데이터 수:", len(df))
    print("train:", len(train_df), "valid:", len(valid_df), "test:", len(test_df))

    return train_df, valid_df, test_df


# =========================
# 5. metric 함수
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# =========================
# 6. 메인
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("사용 디바이스:", device)

    # 1) 데이터
    train_df, valid_df, test_df = load_data(CSV_PATH)

    # 2) 토크나이저 & 모델
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.to(device)

    # 3) Dataset
    train_dataset = EmergencyDataset(
        train_df["text"].tolist(),
        train_df["emergency"].tolist(),
        tokenizer,
    )
    valid_dataset = EmergencyDataset(
        valid_df["text"].tolist(),
        valid_df["emergency"].tolist(),
        tokenizer,
    )
    test_dataset = EmergencyDataset(
        test_df["text"].tolist(),
        test_df["emergency"].tolist(),
        tokenizer,
    )

    # 4) TrainingArguments (구버전 호환용 최소 옵션만 사용)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        num_train_epochs=5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        learning_rate=2e-5,
        weight_decay=0.01,

        logging_steps=50,   # 50 step마다 로그
        save_steps=500,     # 500 step마다 체크포인트 저장
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 5) 학습
    trainer.train()

    # 6) 테스트셋 성능
    print("=== Test 성능 평가 ===")
    test_metrics = trainer.evaluate(test_dataset)
    print(test_metrics)

    # 7) 저장
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"모델과 토크나이저가 저장되었습니다 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
