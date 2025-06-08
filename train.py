import os
import numpy as np
import pickle
from tqdm import tqdm
from audio import AudioFeature
from model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def parse_playlist_file(playlist_path):
    """Đọc playlist có định dạng: <file_path> <genre> <mood>"""
    audio_data = []
    with open(playlist_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"⚠️ Bỏ qua dòng không hợp lệ: {line}")
                continue
            path = " ".join(parts[:-2])  # Tên file có thể chứa dấu cách
            genre = parts[-2]
            mood = parts[-1]
            if os.path.exists(path):
                audio_data.append((path, genre, mood))
            else:
                print(f"❌ File không tồn tại: {path}")
    return audio_data

def extract_features(audio_data):
    """Trích xuất đặc trưng từ các file audio"""
    print("🔍 Đang trích xuất đặc trưng từ các file audio...")
    audio_features = []

    for path, genre, mood in tqdm(audio_data):
        try:
            audio = AudioFeature(path, genre)
            audio.extract_features("mfcc", "chroma", "zcr", "spectral_contrast", "rolloff", "tempo")
            audio_features.append((audio, mood))
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý file {path}: {str(e)}")
            continue

    return audio_features

def prepare_training_data(audio_features):
    print("📊 Đang chuẩn bị dữ liệu huấn luyện...")
    feature_matrix = np.vstack([audio.features for audio, _ in audio_features])
    genre_labels = [audio.genre for audio, _ in audio_features]
    mood_labels = [mood for _, mood in audio_features]
    return feature_matrix, (genre_labels, mood_labels)

def train_model(feature_matrix, labels):
    print("🎯 Đang huấn luyện mô hình...")

    model_cfg = dict(
        tt_test_dict=dict(shuffle=True, test_size=0.2),
        tt_val_dict=dict(shuffle=True, test_size=0.2),
        scaler=StandardScaler(copy=True),
        base_model=RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        param_grid=dict(
            model__criterion=["entropy", "gini"],
            model__max_features=["sqrt", "log2"],
            model__min_samples_leaf=[2, 3, 4],
            model__max_depth=[15, 20, 25]
        ),
        grid_dict=dict(n_jobs=-1, refit=True, scoring="balanced_accuracy", verbose=2),
        kf_dict=dict(n_splits=5, random_state=42, shuffle=True)
    )

    model = Model(feature_matrix, labels, model_cfg)
    model.train_kfold()
    print("\n📈 Kết quả trên tập validation:")
    model.predict(holdout_type="val")
    print("\n📈 Kết quả trên tập test:")
    model.predict(holdout_type="test")

    return model

def save_model(model, output_path="trained_model.pkl"):
    print(f"💾 Đang lưu mô hình vào {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Đã lưu mô hình thành công!")

def main():
    print("🎵 HUẤN LUYỆN MÔ HÌNH (GENRE + MOOD) 🎵")

    playlist_path = "data/playlist.txt"  # ← sửa tên file nếu bạn đặt khác

    audio_data = parse_playlist_file(playlist_path)
    print(f"📁 Tìm thấy {len(audio_data)} bản ghi âm hợp lệ")

    audio_features = extract_features(audio_data)
    print(f"✨ Đã trích xuất đặc trưng từ {len(audio_features)} file")

    feature_matrix, labels = prepare_training_data(audio_features)

    model = train_model(feature_matrix, labels)

    save_model(model)

    print("\n🎉 Quá trình huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()
