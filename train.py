import os
import numpy as np
import pickle
from tqdm import tqdm
from audio import AudioFeature
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, make_scorer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def parse_playlist_file(playlist_path):
    audio_data = []
    with open(playlist_path, "r", encoding="utf-8") as f:
        next(f)  # Bỏ qua dòng header đầu tiên
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"⚠️ Bỏ qua dòng không hợp lệ: {line}")
                continue
            path = " ".join(parts[:-2])
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

def multioutput_balanced_accuracy(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(balanced_accuracy_score(y_true[:, i], y_pred[:, i]))
    return sum(scores) / len(scores)

def train_model(feature_matrix, labels):
    print("🎯 Đang huấn luyện mô hình...")

    y_genre, y_mood = labels

    # Encode labels sang số
    le_genre = LabelEncoder()
    le_mood = LabelEncoder()

    y_genre_enc = le_genre.fit_transform(y_genre)
    y_mood_enc = le_mood.fit_transform(y_mood)

    y_multi = np.vstack([y_genre_enc, y_mood_enc]).T  # (n_samples, 2)

    # Khởi tạo multilabel stratified kfold
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pipeline gồm scaler + RF
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])

    param_grid = {
        'model__criterion': ["entropy", "gini"],
        'model__max_features': ["sqrt", "log2"],
        'model__min_samples_leaf': [2, 3, 4],
        'model__max_depth': [15, 20, 25]
    }

    custom_scorer = make_scorer(multioutput_balanced_accuracy)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=mskf,
        scoring=custom_scorer,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    grid_search.fit(feature_matrix, y_multi)

    print(f"\n✅ Kết quả tốt nhất trên validation:")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best score (balanced accuracy): {grid_search.best_score_:.4f}")

    # Trả về model và các encoder để dùng cho dự đoán sau này
    return grid_search, le_genre, le_mood

def save_model(model_data, output_path="trained_model.pkl"):
    print(f"💾 Đang lưu mô hình vào {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    print("✅ Đã lưu mô hình thành công!")

def main():
    print("🎵 HUẤN LUYỆN MÔ HÌNH (GENRE + MOOD) 🎵")

    playlist_path = "data/playlist.txt"  # ← sửa tên file nếu bạn đặt khác

    audio_data = parse_playlist_file(playlist_path)
    print(f"📁 Tìm thấy {len(audio_data)} bản ghi âm hợp lệ")

    audio_features = extract_features(audio_data)
    print(f"✨ Đã trích xuất đặc trưng từ {len(audio_features)} file")

    feature_matrix, labels = prepare_training_data(audio_features)

    model, le_genre, le_mood = train_model(feature_matrix, labels)

    # Lưu model kèm encoder (dùng khi predict)
    save_model((model, le_genre, le_mood))

    print("\n🎉 Quá trình huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()
