import os
import cv2
import numpy as np

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
RAW_LABEL_DIR = "./vintext_raw/labels/"
RAW_IMG_DIRS = [
    "./vintext_raw/train_images/",
    "./vintext_raw/test_image/",
    "./vintext_raw/unseen_test_images/"
]
OUT_IMG_DIR = "./vintext_svtrv2/images/"
OUT_LABEL_FILE = "./vintext_svtrv2/rec_gt_train.txt"

os.makedirs(OUT_IMG_DIR, exist_ok=True)


# ==========================================
# 2. HÀM NẮN THẲNG ẢNH (PERSPECTIVE TRANSFORM)
# ==========================================
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

    if maxWidth == 0 or maxHeight == 0:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# ==========================================
# 3. TIẾN HÀNH XỬ LÝ VỚI TÌM KIẾM THÔNG MINH
# ==========================================
print("Bắt đầu quét và cắt ảnh VinText...")
processed_count = 0
error_no_img = 0
error_cv2_read = 0
error_parse = 0

label_files = [f for f in os.listdir(RAW_LABEL_DIR) if f.endswith('.txt')]
print(f"[*] Tìm thấy {len(label_files)} file nhãn gốc.")

with open(OUT_LABEL_FILE, 'w', encoding='utf-8') as f_out:
    for label_name in label_files:
        base_name = label_name.replace('.txt', '')
        clean_name = base_name.replace('gt_', '')  # Bỏ tiền tố gt_

        # Danh sách các khả năng tên file ảnh có thể xảy ra trong VinText
        possible_img_names = [
            f"{base_name}.jpg",  # Vd: gt_461.jpg (Nếu có)
            f"{clean_name}.jpg",  # Vd: 461.jpg hoặc im0001.jpg (Phổ biến nhất)
            f"im{clean_name.zfill(4)}.jpg",  # Vd: 461 -> im0461.jpg
            f"{clean_name}.JPG"  # Đề phòng viết hoa đuôi file
        ]

        img_path = None
        target_img_name = ""

        # Quét qua 3 thư mục ảnh với từng khả năng tên file
        for img_name_guess in possible_img_names:
            for img_dir in RAW_IMG_DIRS:
                temp_path = os.path.join(img_dir, img_name_guess)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    target_img_name = img_name_guess
                    break
            if img_path:  # Nếu tìm thấy, thoát vòng lặp tìm tên
                break

        if img_path is None:
            error_no_img += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            error_cv2_read += 1
            continue

        label_path = os.path.join(RAW_LABEL_DIR, label_name)
        with open(label_path, 'r', encoding='utf-8-sig') as f_in:
            lines = f_in.readlines()

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 9:
                error_parse += 1
                continue

            text = ",".join(parts[8:]).strip()

            if text == "###":  # Bỏ qua chữ mờ
                continue

            try:
                coords = list(map(int, parts[:8]))
                pts = np.array(coords).reshape(4, 2)

                # Cắt và nắn thẳng ảnh
                warped_img = four_point_transform(img, pts)

                if warped_img is None or warped_img.shape[0] < 8 or warped_img.shape[1] < 8:
                    continue

                crop_img_name = f"{target_img_name.split('.')[0]}_{idx}.jpg"
                crop_img_path = os.path.join(OUT_IMG_DIR, crop_img_name)

                # Lưu ảnh và ghi nhãn
                success = cv2.imwrite(crop_img_path, warped_img)
                if success:
                    clean_text = text.replace('\n', '').replace('\t', ' ')
                    f_out.write(f"images/{crop_img_name}\t{clean_text}\n")
                    processed_count += 1

            except Exception as e:
                error_parse += 1

print("\n" + "=" * 40)
print("📊 BÁO CÁO KẾT QUẢ XỬ LÝ")
print("=" * 40)
print(f"✅ Số lượng ảnh chữ (cropped) tạo thành công : {processed_count}")
print(f"❌ Số file nhãn bị bỏ qua do không thấy ảnh : {error_no_img}")
print(f"❌ Số ảnh bị lỗi không thể đọc (OpenCV error) : {error_cv2_read}")
print(f"❌ Số dòng nhãn bị lỗi định dạng (Parse error) : {error_parse}")

if processed_count > 0:
    print("\n🎉 CHÚC MỪNG! Dữ liệu VinText đã được chuyển đổi thành công. Bạn đã sẵn sàng để Fine-tune SVTRv2.")