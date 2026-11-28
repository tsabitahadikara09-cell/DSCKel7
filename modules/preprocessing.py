import cv2
import numpy as np

METHOD_DESCRIPTIONS = {
    "Original": "Citra RGB asli sebagai baseline.",
    "Filtering": "Gaussian smoothing untuk mengurangi noise dan variasi intensitas berfrekuensi tinggi.",
    "Segmentation & Masking": "Threshold Otsu menghasilkan mask sederhana; mask diterapkan pada citra untuk menonjolkan area foreground.",
    "Enhancement": "CLAHE pada kanal luminance LAB untuk meningkatkan kontras lokal tanpa mengubah kanal warna secara langsung.",
    "Hair Removal": "Black-hat morphology mendeteksi kandidat rambut gelap, kemudian inpainting mengisi area kandidat.",
    "Grayscale": "Citra diubah menjadi grayscale lalu dikonversi kembali ke 3 kanal agar kompatibel dengan model CNN RGB."
}

def preprocess_image(image, method):
    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if method == "Original":
        return arr
    if method == "Filtering":
        return cv2.GaussianBlur(arr, (3,3), 0)
    if method == "Segmentation & Masking":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masked = cv2.bitwise_and(arr, arr, mask=mask)
        return masked
    if method == "Enhancement":
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    if method == "Hair Removal":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    if method == "Grayscale":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return arr
