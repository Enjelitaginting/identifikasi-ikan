import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import os
import glob

# Fungsi ekstraksi fitur dari gambar ikan
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Hu Moments (log scale)
    hu_moments = cv2.HuMoments(cv2.moments(largest_contour)).flatten()
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-10)
    
    # Shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x,y,w,h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w)/h
    extent = float(area)/(w*h)
    
    # Mean warna di mask contour (BGR)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask)[:3]
    
    features = np.hstack([hu_moments, area, perimeter, aspect_ratio, extent, mean_val])
    return features

# Fungsi load dataset dari folder struktur: dataset/<label>/*.jpg
def load_dataset(path="dataset"):
    X = []
    y = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if os.path.isdir(label_path):
            for img_file in glob.glob(os.path.join(label_path, "*.jpg")):
                img = cv2.imread(img_file)
                if img is None:
                    continue
                fitur = extract_features(img)
                if fitur is not None:
                    X.append(fitur)
                    y.append(label)
    return np.array(X), np.array(y)

# Training model dengan GridSearch SVM untuk hyperparameter tuning
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(svm.SVC(probability=True), param_grid, cv=5, verbose=0)
    grid.fit(X_scaled, y)
    
    best_clf = grid.best_estimator_
    return scaler, best_clf, grid.best_params_

# ===== Streamlit App =====
st.title("Identifikasi Ikan Lokal Pelabuhan Dompak dengan SVM")

st.write("### Training model dari dataset lokal")
if st.button("Mulai Training"):
    with st.spinner("Loading dataset dan training model..."):
        X, y = load_dataset("dataset")
        if len(X) == 0:
            st.error("Dataset kosong atau tidak ditemukan. Pastikan folder dataset/<label>/*.jpg ada.")
        else:
            scaler, clf, best_params = train_model(X, y)
            st.success(f"Training selesai! Best params: {best_params}")
            # Simpan model dan scaler untuk prediksi berikutnya
            import pickle
            with open("model_svm.pkl", "wb") as f:
                pickle.dump(clf, f)
            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

# Upload gambar untuk prediksi
uploaded_file = st.file_uploader("Upload gambar ikan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", caption="Gambar Ikan")

    # Load model dan scaler
    import pickle
    try:
        with open("model_svm.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        st.warning("Model belum dilatih. Silakan klik tombol 'Mulai Training' dulu.")
        clf = None

    if clf is not None:
        fitur = extract_features(img)
        if fitur is not None:
            fitur_scaled = scaler.transform([fitur])
            prediksi = clf.predict(fitur_scaled)
            prob = clf.predict_proba(fitur_scaled).max()
            st.write(f"**Ikan terdeteksi:** {prediksi[0]}")
            st.write(f"**Confidence:** {prob*100:.2f}%")
        else:
            st.write("Gagal mendeteksi ikan pada gambar. Pastikan ikan terlihat jelas dan background tidak rumit.")
