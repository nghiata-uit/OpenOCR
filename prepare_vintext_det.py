import os
import json
import unicodedata

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
RAW_LABEL_DIR = "./vintext_raw/labels/"
RAW_IMG_DIRS = [
    "./vintext_raw/train_images/",
    "./vintext_raw/test_image/",
    "./vintext_raw/unseen_test_images/"
]
# Nơi lưu file nhãn dùng để train Detection
OUT_LABEL_TRAIN = "./vintext_raw/train_det_gt.txt"
OUT_LABEL_TEST = "./vintext_raw/test_det_gt.txt"

def prepare_detection_labels():
    print("Bắt đầu chuyển đổi nhãn VinText sang định dạng DBNet (Detection)...")
    
    label_files = [f for f in os.listdir(RAW_LABEL_DIR) if f.endswith('.txt')]
    print(f"[*] Tìm thấy {len(label_files)} file nhãn gốc.")
    
    train_count = 0
    test_count = 0
    
    with open(OUT_LABEL_TRAIN, 'w', encoding='utf-8') as f_out_train, \
         open(OUT_LABEL_TEST, 'w', encoding='utf-8') as f_out_test:
        
        for label_name in label_files:
            base_name = label_name.replace('.txt', '')
            clean_name = base_name.replace('gt_', '')
            
            # Danh sách các khả năng tên file ảnh
            possible_img_names = [
                f"{base_name}.jpg",
                f"{clean_name}.jpg",
                f"im{clean_name.zfill(4)}.jpg",
                f"{clean_name}.JPG"
            ]
            
            img_path = None
            img_rel_path = ""
            is_train = True
            
            for img_name_guess in possible_img_names:
                for img_dir in RAW_IMG_DIRS:
                    temp_path = os.path.join(img_dir, img_name_guess)
                    if os.path.exists(temp_path):
                        img_path = temp_path
                        # Lưu đường dẫn tương đối từ vintext_raw (vd: train_images/im0123.jpg)
                        img_rel_path = os.path.join(os.path.basename(img_dir.rstrip('/')), img_name_guess)
                        if "test" in img_dir.lower():
                            is_train = False
                        break
                if img_path:
                    break
            
            if img_path is None:
                continue
                
            label_path = os.path.join(RAW_LABEL_DIR, label_name)
            with open(label_path, 'r', encoding='utf-8-sig') as f_in:
                lines = f_in.readlines()
                
            labels = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                    
                text = ",".join(parts[8:]).strip()
                
                try:
                    coords = list(map(int, parts[:8]))
                    points = [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]]
                    ]
                    
                    # Nếu text là ### (chữ mờ), DBNet sẽ tự động phớt lờ nó (ignore)
                    clean_text = text.replace('\n', '').replace('\t', ' ')
                    clean_text = unicodedata.normalize('NFC', clean_text)
                    
                    labels.append({
                        "transcription": clean_text,
                        "points": points
                    })
                except Exception as e:
                    continue
                    
            if not labels:
                continue
                
            # Định dạng OpenOCR: path\t[{"transcription": "...", "points": [...]}]
            json_str = json.dumps(labels, ensure_ascii=False)
            output_line = f"{img_rel_path}\t{json_str}\n"
            
            if is_train:
                f_out_train.write(output_line)
                train_count += 1
            else:
                f_out_test.write(output_line)
                test_count += 1

    print("=" * 40)
    print(f"✅ Đã tạo nhãn TRAIN Detection cho {train_count} ảnh: {OUT_LABEL_TRAIN}")
    print(f"✅ Đã tạo nhãn TEST Detection cho {test_count} ảnh: {OUT_LABEL_TEST}")

if __name__ == "__main__":
    prepare_detection_labels()
