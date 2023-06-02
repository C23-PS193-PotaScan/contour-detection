import cv2
import numpy as np

def preprocess_image(image):
    # Preprocessing gambar
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_contours(image):
    # Deteksi tepi menggunakan Canny edge detection
    _, mask = cv2.threshold(image, 170,30, cv2.THRESH_BINARY)

    # Mencari kontur dalam gambar biner
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_and_crop_images(image, contours):
    # Menggambar kontur pada gambar asli
    cv2.drawContours(image, contours, -50, (0, 0, 255), 2)
    
    return image

def main():
    image_path = './images/IMG-20230514-WA0019.jpg'

    # Membaca gambar dari file
    image = cv2.imread(image_path)

    # Preprocessing gambar
    preprocessed_image = preprocess_image(image)

    # Deteksi kontur
    contours = detect_contours(preprocessed_image)

    # Menggambar kontur dan melakukan crop
    result_image = draw_and_crop_images(image.copy(), contours)

    # Menampilkan gambar dengan kontur
    cv2.imshow('Deteksi Kontur', result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Menjalankan fungsi main
if __name__ == '__main__':
    main()
