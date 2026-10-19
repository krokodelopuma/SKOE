import cv2
import numpy as np
from pathlib import Path
import time

#берет окно 3×3, 255 если в окне есть хотя бы 1 белый
def dilate_native(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    r = ksize // 2
    h, w = img.shape
    out = np.zeros_like(img)
    padded = np.zeros((h + 2 * r, w + 2 * r), dtype=img.dtype)
    padded[r:r + h, r:r + w] = img
    for y in range(h):
        for x in range(w):
            window = padded[y:y + ksize, x:x + ksize]
            out[y, x] = 255 if window.max() > 0 else 0
    return out

#9 сдвигов массива, берет поэлементный максимум
def dilate_native_numpy(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    r = ksize // 2
    h, w = img.shape
    padded = np.zeros((h + 2 * r, w + 2 * r), dtype=img.dtype)
    padded[r:r + h, r:r + w] = img
    out = np.zeros_like(img)
    for dy in range(ksize):
        for dx in range(ksize):
            out = np.maximum(out, padded[dy:dy + h, dx:dx + w])
    return out

#Создает ядро 3×3 из единиц и вызывает встроенную функцию
def dilate_opencv(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

#бинаризация
def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

#генерация тестового изображения
def make_test_image(size: int = 512) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    s = size // 10
    cv2.rectangle(img, (s, s), (4 * s, 4 * s), 255, -1)
    cv2.circle(img, (7 * s, 3 * s), s + s // 2, 255, -1)
    cv2.line(img, (s, 8 * s), (9 * s, 8 * s), 255, 3)
    return img

#сравнение изображений
def compare(a: np.ndarray, b: np.ndarray) -> bool:
    return cv2.absdiff(a, b).max() == 0

#сравнение быстродействия
def benchmark(func, img, ksize=3, repeats=1):
    t0 = time.perf_counter()
    for _ in range(repeats):
        res = func(img, ksize)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats, res


def main():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    binary = make_test_image(512)

    sample = Path("sample.jpg")
    if sample.exists():
        gray = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        binary = binarize_otsu(gray)

    d_native = dilate_native(binary, 3)
    d_native_np = dilate_native_numpy(binary, 3)
    d_cv = dilate_opencv(binary, 3)

    print("Python-цикл (native) == OpenCV :", compare(d_native, d_cv))
    print("NumPy (native)       == OpenCV :", compare(d_native_np, d_cv))

    cv2.imwrite(str(out_dir / "1_source.png"), binary)
    cv2.imwrite(str(out_dir / "2_dilate_openCV.png"), d_cv)
    cv2.imwrite(str(out_dir / "3_dilate_native.png"), d_native)
    cv2.imwrite(str(out_dir / "4_dilate_native_numpy.png"), d_native_np)

    print("\nСравнение быстродействия (изображение {}x{})".format(*binary.shape))
    t_py, _ = benchmark(dilate_native, binary, 3, repeats=1)
    t_np, _ = benchmark(dilate_native_numpy, binary, 3, repeats=10)
    t_cv, _ = benchmark(dilate_opencv, binary, 3, repeats=50)

    print(f"Python-цикл (native)  : {t_py*1000:10.3f} мс")
    print(f"NumPy (native)        : {t_np*1000:10.3f} мс")
    print(f"OpenCV cv2.dilate     : {t_cv*1000:10.3f} мс")

    try:
        cv2.imshow("Source", binary)
        cv2.imshow("OpenCV dilation", d_cv)
        cv2.imshow("Native dilation", d_native)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
