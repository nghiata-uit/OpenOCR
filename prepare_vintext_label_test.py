import os
import random

# Cấu hình đường dẫn
DATA_DIR = "./vintext_svtrv2"
ALL_LABEL_FILE = os.path.join(DATA_DIR, "rec_gt_train.txt")
TEMP_TRAIN_FILE = os.path.join(DATA_DIR, "rec_gt_train_temp.txt")
TEST_FILE = os.path.join(DATA_DIR, "rec_gt_test.txt")

# Tỷ lệ chia (Ví dụ: 90% Train, 10% Test)
TRAIN_RATIO = 0.9

print("Đang tiến hành chia tách tập dữ liệu...")

# 1. Đọc toàn bộ nhãn hiện có
if not os.path.exists(ALL_LABEL_FILE):
    print(f"❌ Không tìm thấy file {ALL_LABEL_FILE}. Hãy chắc chắn bạn đã chạy kịch bản cắt ảnh thành công.")
else:
    with open(ALL_LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Dọn dẹp các dòng trống (nếu có)
    lines = [line for line in lines if line.strip()]
    total_samples = len(lines)

    print(f"[*] Tổng số ảnh chữ đang có: {total_samples}")

    # 2. Xáo trộn ngẫu nhiên dữ liệu
    # Cố định random seed (42) để nếu bạn chạy lại nhiều lần, cách chia vẫn không bị thay đổi
    random.seed(42)
    random.shuffle(lines)

    # 3. Tính toán mốc chia cắt
    split_idx = int(total_samples * TRAIN_RATIO)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    # 4. Ghi dữ liệu ra các file mới
    # Ghi file Test
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    # Ghi đè file Train bằng dữ liệu đã cắt bớt
    with open(TEMP_TRAIN_FILE, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # Thay thế file gốc bằng file Train mới
    os.replace(TEMP_TRAIN_FILE, ALL_LABEL_FILE)

    print("\n" + "=" * 40)
    print("🎉 CHIA TÁCH DATASET THÀNH CÔNG!")
    print("=" * 40)
    print(f"📁 Tập Huấn luyện (Train) : {len(train_lines)} mẫu (Lưu tại: rec_gt_train.txt)")
    print(f"📁 Tập Đánh giá (Test)   : {len(test_lines)} mẫu (Lưu tại: rec_gt_test.txt)")