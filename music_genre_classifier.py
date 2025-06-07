# music_genre_classifier.py

import numpy as np
import pandas as pd
import os
import pickle
from audio import AudioFeature
from model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def parse_audio_playlist(playlist):
    df = pd.read_csv(playlist, sep="\t")
    df = df[["Location", "Genre"]]
    paths = df["Location"].values.astype(str)
    paths = np.char.replace(paths, "Macintosh HD", "")
    genres = df["Genre"].values
    return zip(paths, genres)

def train_model():
    print("🔧 Đang huấn luyện mô hình...")
    all_metadata = parse_audio_playlist("data/Subset.txt")
    audio_features = []

    for path, genre in all_metadata:
        audio = AudioFeature(path, genre)
        audio.extract_features("mfcc", "chroma", "zcr", "spectral_contrast", "rolloff", "tempo", save_local=False)
        audio_features.append(audio)

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    genre_labels = [audio.genre for audio in audio_features]

    model_cfg = dict(
        tt_test_dict=dict(shuffle=True, test_size=0.3),
        tt_val_dict=dict(shuffle=True, test_size=0.25),
        scaler=StandardScaler(copy=True),
        base_model=RandomForestClassifier(
            random_state=42,
            n_jobs=4,
            class_weight="balanced",
            n_estimators=250,
            bootstrap=True,
        ),
        param_grid=dict(
            model__criterion=["entropy", "gini"],
            model__max_features=["log2", "sqrt"],
            model__min_samples_leaf=np.arange(2, 4),
        ),
        grid_dict=dict(n_jobs=4, refit=True, scoring="balanced_accuracy"),
        kf_dict=dict(n_splits=3, random_state=42, shuffle=True),
    )

    model = Model(feature_matrix, genre_labels, model_cfg)
    model.train_kfold()
    model.predict(holdout_type="val")
    model.predict(holdout_type="test")

    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Đã lưu mô hình vào: trained_model.pkl")

    return model

def load_model():
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("📦 Mô hình đã được nạp từ file.")
    return model

def predict_genre(model, mp3_path):
    audio = AudioFeature(mp3_path, genre=None)
    audio.extract_features("mfcc", "chroma", "zcr", "spectral_contrast", "rolloff", "tempo")
    X_new = model.best_estimator['scaler'].transform(audio.features.reshape(1, -1))
    pred_index = model.best_estimator['model'].predict(X_new)[0]
    genre = model.encoder.inverse_transform([pred_index])[0]
    print(f"\n🎵 File '{mp3_path}' được dự đoán là thể loại: {genre}")
    return genre

if __name__ == "__main__":
    print("🎧 HỆ THỐNG PHÂN LOẠI THỂ LOẠI NHẠC 🎶")
    choice = input("👉 Bạn muốn:\n1. Huấn luyện lại mô hình\n2. Dự đoán bằng mô hình đã có\nChọn (1 hoặc 2): ")

    if choice == "1":
        model = train_model()
    elif choice == "2":
        model = load_model()
    else:
        print("❌ Lựa chọn không hợp lệ.")
        exit()

    mp3_path = input("🔍 Nhập đường dẫn đến file nhạc (.mp3): ")
    if not os.path.exists(mp3_path):
        print("❌ Không tìm thấy file mp3. Kiểm tra lại đường dẫn.")
    else:
        predict_genre(model, mp3_path)
