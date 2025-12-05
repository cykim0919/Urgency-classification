from emergency_inference import predict_emergency

texts = [
    "기숙사에서 타는 냄새가 나요",
    "연기가 조금 납니다",
    "불꽃이 보여요",
    "수도에서 물이 안 나와요"
]

for t in texts:
    print(t, "→", predict_emergency(t))
