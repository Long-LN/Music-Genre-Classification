import sys
import pickle
from audio import AudioFeature

# Đường dẫn tới file model đã huấn luyện
MODEL_PATH = "trained_model.pkl"

def predict(file_path):
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Trích xuất đặc trưng từ file âm thanh
    print(f"🎵 Đang phân tích file: {file_path}")
    audio = AudioFeature(file_path, genre="unknown")  # genre chỉ là placeholder
    audio.extract_features("mfcc", "chroma", "zcr", "spectral_contrast", "rolloff", "tempo")

    # Dự đoán
    predictions = model.predict(audio.features.reshape(1, -1))
    genre, mood = predictions[0][0], predictions[1][0]
    print(f"🔍 Kết quả dự đoán:\n   ➤ Thể loại: {genre}\n   ➤ Tâm trạng: {mood}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Vui lòng cung cấp đường dẫn file âm thanh để dự đoán.")
        print("Ví dụ: python predict.py audio/test1.mp3")
        sys.exit(1)

    filepath = sys.argv[1]
    predict(filepath)
