import os
import cv2
import numpy as np

# Cấu hình đường dẫn
RAW_IMG_DIR = "./vintext_raw/images/"
RAW_LABEL_DIR = "./vintext_raw/labels/"
OUT_IMG_DIR = "./vintext_svtrv2/images/"
OUT_LABEL_FILE = "./vintext_svtrv2/rec_gt_train.txt"

os.makedirs(OUT_IMG_DIR, exist_ok=True)


def order_points(pts):
    # Khởi tạo danh sách tọa độ nắn thẳng [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính toán chiều rộng và chiều cao ảnh mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


print("Bắt đầu xử lý dữ liệu VinText...")
with open(OUT_LABEL_FILE, 'w', encoding='utf-8') as f_out:
    for label_name in os.listdir(RAW_LABEL_DIR):
        img_name = label_name.replace('.txt', '.jpg')
        img_path = os.path.join(RAW_IMG_DIR, img_name)
        label_path = os.path.join(RAW_LABEL_DIR, label_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        with open(label_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        for idx, line in enumerate(lines):
            # VinText phân cách bằng dấu phẩy
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue

            text = ",".join(parts[8:])  # Ghép lại nếu chữ có chứa dấu phẩy

            # VinText dùng "###" để đánh dấu các chữ mờ không thể đọc được
            if text == "###":
                continue

            try:
                coords = list(map(int, parts[:8]))
                pts = np.array(coords).reshape(4, 2)

                # Cắt và nắn thẳng ảnh
                warped_img = four_point_transform(img, pts)

                # Bỏ qua các ảnh quá nhỏ hoặc bị lỗi
                if warped_img.shape[0] < 5 or warped_img.shape[1] < 5:
                    continue

                crop_img_name = f"{img_name.split('.')[0]}_{idx}.jpg"
                crop_img_path = os.path.join(OUT_IMG_DIR, crop_img_name)

                cv2.imwrite(crop_img_path, warped_img)

                # Ghi vào file chuẩn OpenOCR (dùng \t)
                clean_text = text.replace('\n', '').replace('\t', ' ')
                f_out.write(f"{crop_img_name}\t{clean_text}\n")

            except Exception as e:
                pass  # Bỏ qua các dòng lỗi tọa độ

print("Hoàn tất! Dữ liệu đã sẵn sàng cho SVTRv2.")