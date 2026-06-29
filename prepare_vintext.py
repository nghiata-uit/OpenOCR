import os
import cv2
import numpy as np

# 1. CẤU HÌNH ĐƯỜNG DẪN DỰA TRÊN CẤU TRÚC THỰC TẾ
RAW_LABEL_DIR = "./vintext_raw/labels/"
RAW_IMG_DIRS = [
    "./vintext_raw/train_images/",
    "./vintext_raw/test_image/",
    "./vintext_raw/unseen_test_images/"
]
OUT_IMG_DIR = "./vintext_svtrv2/images/"
OUT_LABEL_FILE = "./vintext_svtrv2/rec_gt_train.txt"

os.makedirs(OUT_IMG_DIR, exist_ok=True)


# 2. HÀM NẮN THẲNG ẢNH (PERSPECTIVE TRANSFORM)
def order_points(pts):
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


# 3. TIẾN HÀNH XỬ LÝ
print("Bắt đầu quét và cắt ảnh VinText...")
processed_count = 0

with open(OUT_LABEL_FILE, 'w', encoding='utf-8') as f_out:
    for label_name in os.listdir(RAW_LABEL_DIR):
        if not label_name.endswith('.txt'):
            continue

        img_name = label_name.replace('.txt', '.jpg')
        if img_name.startswith('gt_'):
            img_name = img_name[3:]
        elif img_name.startswith('res_'):
            img_name = img_name[4:]
            
        img_path = None

        # Quét qua 3 thư mục ảnh để tìm ảnh tương ứng với nhãn
        for img_dir in RAW_IMG_DIRS:
            temp_path = os.path.join(img_dir, img_name)
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        # Nếu vẫn không thấy ảnh (bị thất lạc), bỏ qua file nhãn này
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        label_path = os.path.join(RAW_LABEL_DIR, label_name)
        with open(label_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue

            text = ",".join(parts[8:])

            # Bỏ qua các chữ bị mờ/không đọc được (được gán nhãn ###)
            if text == "###":
                continue

            try:
                coords = list(map(int, parts[:8]))
                pts = np.array(coords).reshape(4, 2)

                # Cắt và nắn thẳng ảnh
                warped_img = four_point_transform(img, pts)

                # Bỏ qua các ảnh quá bé (nhiễu)
                if warped_img.shape[0] < 8 or warped_img.shape[1] < 8:
                    continue

                crop_img_name = f"{img_name.split('.')[0]}_{idx}.jpg"
                crop_img_path = os.path.join(OUT_IMG_DIR, crop_img_name)

                # Lưu ảnh chữ đã cắt
                cv2.imwrite(crop_img_path, warped_img)

                # Ghi vào file nhãn chuẩn OpenOCR
                clean_text = text.replace('\n', '').replace('\t', ' ')
                f_out.write(f"images/{crop_img_name}\t{clean_text}\n")
                processed_count += 1

            except Exception as e:
                pass  # Bỏ qua các dòng lỗi tọa độ

print(f"✅ Hoàn tất! Đã tạo thành công {processed_count} ảnh chữ cắt (cropped images).")
print(f"📁 Dữ liệu được lưu tại: {OUT_IMG_DIR}")
print(f"📄 File nhãn được lưu tại: {OUT_LABEL_FILE}")