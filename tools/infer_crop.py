import os
import sys
import argparse
import cv2
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.infer_det import OpenDetector
from tools.infer_rec import OpenRecognizer
from tools.engine.config import Config
from tools.utils.utility import get_image_file_list
from tools.utils.logging import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference DBNet + SVTRv2 on cropped word images")
    parser.add_argument("--img", type=str, required=True, help="Path to cropped image or folder of cropped images")
    parser.add_argument("--det_config", type=str, default="configs/det/dbnet/dbnet_vintext.yml", help="Path to DBNet config")
    parser.add_argument("--det_model", type=str, default="./output/det_repsvtr_db/best.pth", help="Path to DBNet model weights")
    parser.add_argument("--rec_config", type=str, default="configs/rec/svtrv2/svtrv2_vn.yml", help="Path to SVTRv2 config")
    parser.add_argument("--rec_model", type=str, default="./best.pth", help="Path to SVTRv2 model weights")
    parser.add_argument("--pad", type=int, default=6, help="Padding margin around word crop")
    return parser.parse_args()


def four_point_transform(image, pts, pad=6):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 5 or maxHeight <= 5:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if pad > 0 and warped is not None:
        warped = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return warped


def main():
    args = parse_args()

    # 1. Khởi tạo DBNet
    logger.info(f"Loading DBNet detector from {args.det_model}...")
    det_cfg = Config(args.det_config).cfg
    if os.path.exists(args.det_model):
        det_cfg["Global"]["pretrained_model"] = args.det_model
    det_model = OpenDetector(det_cfg)

    # 2. Khởi tạo SVTRv2
    logger.info(f"Loading SVTRv2 recognizer from {args.rec_model}...")
    rec_cfg = Config(args.rec_config).cfg
    if os.path.exists(args.rec_model):
        rec_cfg["Global"]["pretrained_model"] = args.rec_model
    rec_model = OpenRecognizer(rec_cfg)

    # 3. Lấy danh sách ảnh
    image_list = get_image_file_list(args.img)
    logger.info(f"Tìm thấy {len(image_list)} ảnh để suy luận.")

    print("\n" + "=" * 65)
    print(f"{'STT':<5} | {'Ảnh đầu vào':<25} | {'Từ nhận dạng':<15} | {'Độ tự tin'}")
    print("-" * 65)

    for idx, img_file in enumerate(image_list):
        orig_img = cv2.imread(img_file)
        if orig_img is None:
            continue

        # Chạy DBNet để tinh chỉnh viền chữ
        det_res = det_model(img_path=img_file)[0]
        boxes = det_res.get("boxes", [])

        if len(boxes) > 0:
            # Chọn box có diện tích lớn nhất trong ảnh cắt sẵn
            best_box = max(boxes, key=lambda b: cv2.contourArea(np.array(b, dtype=np.float32)))
            crop_img = four_point_transform(orig_img, best_box, pad=args.pad)
            if crop_img is None:
                crop_img = cv2.copyMakeBorder(orig_img, args.pad, args.pad, args.pad, args.pad, cv2.BORDER_REPLICATE)
        else:
            # Nếu DBNet không tìm thấy (do ảnh đã quá khít chữ), dùng trực tiếp ảnh gốc có viền đệm
            crop_img = cv2.copyMakeBorder(orig_img, args.pad, args.pad, args.pad, args.pad, cv2.BORDER_REPLICATE)

        # Chạy SVTRv2 nhận dạng
        rec_res = rec_model(img_numpy=crop_img)[0]
        text = rec_res["text"]
        score = rec_res["score"]

        file_name = os.path.basename(img_file)
        print(f"{idx + 1:<5} | {file_name:<25} | {text:<15} | {score * 100:6.2f}%")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
