
import cv2
import numpy as np


# =====================================================
# METHOD DESCRIPTIONS
# =====================================================

METHOD_DESCRIPTIONS = {
    "Original": "Citra RGB asli sebagai pembanding.",

    "Filtering": "Mengurangi noise menggunakan Mean, Median, atau Gaussian Filter.",

    "Mean Filter": "Mengurangi noise dengan menghitung rata-rata intensitas piksel di sekitar piksel.",
    "Median Filter": "Mengurangi noise menggunakan nilai median piksel tetangga, terutama untuk salt-and-pepper noise.",
    "Gaussian Filter": "Mengurangi noise menggunakan Gaussian smoothing.",

    "Segmentation & Masking": "Memisahkan lesi kulit menjadi area putih dan background hitam menggunakan thresholding dan morphological processing.",

    "Enhancement": "Meningkatkan kontras lokal menggunakan CLAHE pada kanal luminance LAB.",

    "Hair Removal": "Mendeteksi rambut menggunakan Black Hat Morphology, menghasilkan hair mask, kemudian menghilangkan rambut menggunakan inpainting.",

    "Grayscale": "Mengubah citra RGB menjadi grayscale."
}


# =====================================================
# IMAGE CONVERSION
# =====================================================

def to_rgb(image):
    """
    Mengubah input PIL atau NumPy menjadi RGB uint8.
    """

    if hasattr(image, "convert"):
        image = image.convert("RGB")

    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    if arr.dtype != np.uint8:
        if arr.max() <= 1:
            arr = arr * 255

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


# =====================================================
# 1. FILTERING
# =====================================================

def apply_filter(image, filter_type="Gaussian Filter", kernel_size=5):

    arr = to_rgb(image)

    # Kernel harus ganjil
    kernel_size = max(3, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    if filter_type in ["Mean Filter", "Mean"]:
        return cv2.blur(
            arr,
            (kernel_size, kernel_size)
        )

    elif filter_type in ["Median Filter", "Median"]:
        return cv2.medianBlur(
            arr,
            kernel_size
        )

    elif filter_type in ["Gaussian Filter", "Gaussian"]:
        return cv2.GaussianBlur(
            arr,
            (kernel_size, kernel_size),
            0
        )

    return arr


# =====================================================
# 2. SEGMENTATION & MASKING
# =====================================================

def segment_image(image):

    arr = to_rgb(image)

    # RGB ke grayscale
    gray = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2GRAY
    )

    # Mengurangi noise sebelum thresholding
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu inverse:
    # Area gelap (kandidat lesi) menjadi putih
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological closing untuk menyambungkan
    # area lesi yang terputus
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # Menghilangkan noise kecil
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    # Memilih komponen terbesar sebagai kandidat lesi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if count > 1:

        component_areas = stats[1:, cv2.CC_STAT_AREA]

        largest_index = np.argmax(component_areas) + 1

        largest_area = stats[
            largest_index,
            cv2.CC_STAT_AREA
        ]

        total_area = mask.shape[0] * mask.shape[1]

        # Hindari memilih seluruh background
        # jika threshold menghasilkan area terlalu besar
        if largest_area < total_area * 0.65:

            clean_mask = np.zeros_like(mask)

            clean_mask[
                labels == largest_index
            ] = 255

            mask = clean_mask

    # Mask diterapkan ke citra asli
    segmented = cv2.bitwise_and(
        arr,
        arr,
        mask=mask
    )

    return mask, segmented


# =====================================================
# 3. HAIR REMOVAL
# =====================================================

def hair_removal_steps(image):

    arr = to_rgb(image)

    # Tahap 1: Grayscale
    gray = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2GRAY
    )

    # Tahap 2: Black Hat Morphology
    # Mendeteksi struktur gelap seperti rambut
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Normalisasi visual Black Hat
    blackhat_display = cv2.normalize(
        blackhat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Threshold otomatis menggunakan Otsu
    _, hair_mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Menghilangkan noise kecil
    # sekaligus mempertebal kandidat rambut
    mask_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    hair_mask = cv2.morphologyEx(
        hair_mask,
        cv2.MORPH_OPEN,
        mask_kernel
    )

    hair_mask = cv2.dilate(
        hair_mask,
        mask_kernel,
        iterations=1
    )

    # Tahap 3: Inpainting
    # Mengisi area rambut berdasarkan piksel sekitarnya
    hair_removed = cv2.inpaint(
        arr,
        hair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return {
        "original": arr,
        "blackhat": blackhat_display,
        "mask": hair_mask,
        "hair_removed": hair_removed
    }


# =====================================================
# 4. ENHANCEMENT
# =====================================================

def enhance_image(image):

    arr = to_rgb(image)

    lab = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced = cv2.merge([
        enhanced_l,
        a,
        b
    ])

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2RGB
    )


# =====================================================
# 5. GRAYSCALE
# =====================================================

def grayscale_image(image):

    arr = to_rgb(image)

    gray = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2GRAY
    )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )


# =====================================================
# MAIN PREPROCESSING FUNCTION
# =====================================================

def preprocess_image(
    image,
    method,
    filter_type="Gaussian Filter",
    kernel_size=5
):

    arr = to_rgb(image)

    if method == "Original":
        return arr

    elif method in [
        "Filtering",
        "Mean Filter",
        "Median Filter",
        "Gaussian Filter"
    ]:

        selected_filter = (
            filter_type if method == "Filtering"
            else method
        )

        return apply_filter(
            arr,
            selected_filter,
            kernel_size
        )

    elif method == "Segmentation & Masking":

        _, segmented = segment_image(arr)

        return segmented

    elif method == "Enhancement":

        return enhance_image(arr)

    elif method == "Hair Removal":

        return hair_removal_steps(arr)["hair_removed"]

    elif method == "Grayscale":

        return grayscale_image(arr)

    return arr
