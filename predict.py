import sys
import pickle
import numpy as np
from audio import AudioFeature
from sklearn.metrics import balanced_accuracy_score


MODEL_PATH = "trained_model.pkl"

def multioutput_balanced_accuracy(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(balanced_accuracy_score(y_true[:, i], y_pred[:, i]))
    return sum(scores) / len(scores)

def predict(file_path):
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model, encoder_genre, encoder_mood = model_data
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    try:
        print(f"🎵 Đang phân tích file: {file_path}")
        audio = AudioFeature(file_path, genre="unknown")
        audio.extract_features("mfcc", "chroma", "zcr", "spectral_contrast", "rolloff", "tempo")
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất đặc trưng: {e}")
        return

    try:
        # GridSearchCV không có predict_single, chỉ có predict
        # Bạn dùng predict với dữ liệu chuẩn hóa tự động bởi pipeline
        y_pred = model.predict(audio.features.reshape(1, -1))  # trả về 2 nhãn cùng lúc

        # y_pred là 2D array, mỗi cột là 1 output
        genre_idx = y_pred[0, 0]
        mood_idx = y_pred[0, 1]

        genre_name = encoder_genre.inverse_transform([genre_idx])[0]
        mood_name = encoder_mood.inverse_transform([mood_idx])[0]

        print(f"🔍 Kết quả dự đoán:\n   ➤ Thể loại: {genre_name}\n   ➤ Tâm trạng: {mood_name}")
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Vui lòng cung cấp đường dẫn file âm thanh để dự đoán.")
        sys.exit(1)

    filepath = sys.argv[1]
    predict(filepath)
