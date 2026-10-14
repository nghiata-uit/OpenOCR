#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ sinh dữ liệu chữ tiếng Việt tổng hợp (Vietnamese Synthetic Text Generator)
Phục vụ huấn luyện / finetune mô hình OCR (SVTRv2 / CRNN / Master / PARSeq).
Giải quyết triệt để các ca khó:
1. Dấu kép tiếng Việt (Circumflex + Tilde/Hook/Dot: ẫ, ễ, ỗ, ỡ, ử, ẩ, ể, ổ, ở, ặ, ệ, ộ, ợ, ự)
2. Chữ biển hiệu, đèn LED, phát sáng neon, viền chữ, bóng đổ
3. Nền phức tạp: Tôn sóng (corrugated metal), tường gạch, vân gỗ, gradient
4. Nhiễu thực tế: Che khuất (Occlusion), lóa sáng (Glare/Flare), mờ nhòe (Blur), nén JPEG
"""

import os
import sys
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==============================================================================
# 1. KHO TỪ VỰNG TIẾNG VIỆT ĐA DẠNG & TẬP TRUNG VÀO DẤU KÉP KHÓ
# ==============================================================================

# Từ vựng biển hiệu, bảng hiệu đường phố, quán ăn, dịch vụ thực tế tại Việt Nam
REAL_WORLD_WORDS = [
    # Biển hiệu quán ăn, đồ uống
    "TRÀ", "CHANH", "CÀ PHÊ", "TIỆM", "MIỄN", "PHÍ", "BÚN", "PHỞ", "CƠM",
    "BÁNH MÌ", "TẠP HÓA", "BÁNH CUỐN", "NƯỚC MÍA", "BÚN CHẢ", "BÚN ĐẬU", "MẮM TÔM",
    "GÀ RÁN", "HẢI SẢN", "QUÁN ĂN", "ĐẶC SẢN", "SINH TỐ", "NƯỚC ÉP", "CHÈ", "KHOAI LANG",
    "LẨU", "NƯỚNG", "BÒ TƠ", "DÊ NÚI", "THỊT CHÓ", "CHÁO LÒNG", "HỦ TIẾU", "BÁNH CANH",
    "NEM NƯỚNG", "BÁNH XÈO", "BÁNH KHỌT", "ỐC LUỘC", "CƠM TẤM", "CƠM BÌNH DÂN", "CƠM NIÊU",
    # Dịch vụ, sửa chữa, cửa hàng
    "SỬA CHỮA", "XE MÁY", "BẢO HIỂM", "RỬA XE", "CẮT TÓC", "KHOÁNG", "THỜI TRANG",
    "NHA KHOA", "NHÀ THUỐC", "ĐỒNG HỒ", "ÁO CƯỚI", "CHỤP ẢNH", "IN ẤN", "PHOTOCOPY",
    "XĂNG DẦU", "KHÁCH SẠN", "NHÀ NGHỈ", "MASSAGE", "SPA", "THẨM MỸ", "VÀNG BẠC",
    "CẦM ĐỒ", "ĐIỆN THOẠI", "GIA DỤNG", "VẬT LIỆU", "XÂY DỰNG", "CỬA HÀNG", "ĐẠI LÝ",
    "ĐIỆN MÁY", "PHỤ TÙNG", "SẮT THÉP", "SƠN NHÀ", "NỘI THẤT", "GỖ", "NHÔM KÍNH",
    # Từ ngữ quảng cáo, thông tin
    "GIẢM GIÁ", "KHUYẾN MÃI", "KHAI TRƯƠNG", "ĐỒNG GIÁ", "MỞ CỬA", "ĐÓNG CỬA",
    "CHÍNH HÃNG", "UY TÍN", "CHẤT LƯỢNG", "GIÁ RẺ", "HOTLINE", "ĐỊA CHỈ", "CHI NHÁNH",
    "TẦNG", "SỐ", "HÂN HẠNH", "PHỤC VỤ", "KÍNH CHÀO", "QUÝ KHÁCH", "BẢO HÀNH", "LẮP ĐẶT",
    # Họ tên & Địa danh phổ biến
    "NGUYỄN", "TRẦN", "LÊ", "PHẠM", "HOÀNG", "HUỲNH", "PHAN", "VŨ", "VÕ", "ĐẶNG",
    "BÙI", "ĐỖ", "HỒ", "NGÔ", "DƯƠNG", "LÝ", "ĐÀO", "ĐINH", "HÀ NỘI", "SÀI GÒN",
    "ĐÀ NẴNG", "HẢI PHÒNG", "CẦN THƠ", "NHA TRANG", "HUẾ", "QUẢNG NINH", "VŨNG TÀU"
]

# Các từ chứa tổ hợp dấu kép phức tạp (chữ Â, Ă, Ê, Ô, Ơ, Ư kết hợp Ngã, Hỏi, Nặng, Sắc, Huyền)
HARD_DIACRITIC_WORDS = [
    # Dấu ngã trên mũ/móc (~): ẫ, ẵ, ễ, ỗ, ỡ, ử, ỹ, ĩ, ũ
    "MIỄN", "DIỄN", "TIỄN", "VIỄN", "NGUYỄN", "NHUYỄN", "TRIỄN", "NGHỄNH",
    "LỖI", "NỖI", "CỖI", "CHỖ", "DỖ", "GỖ", "HỖ", "LỖ", "MỖ", "RỖ", "SỖ", "THỖ", "VỖ",
    "BẪY", "CẪY", "HẪNG", "NGẪM", "NGẪU", "DẪN", "LẪN", "MẪU", "NẪU", "ẪU",
    "BẴNG", "ĐẴNG", "LẴNG", "VẴN", "ẴNG",
    "CỠ", "MỠ", "NHỠ", "NỠ", "ĐỠ", "BỠ", "GỠ", "HỠ",
    "BỮA", "CHỮA", "NỮA", "SỮA", "VỮA", "LỮ", "NHỮNG", "CỮ", "GIỮ", "DỮ", "VỮNG", "KỸ",
    # Dấu hỏi trên mũ/móc (?): ẩ, ẳ, ể, ổ, ở, ử, ỷ
    "HIỂM", "ĐIỂM", "KIỂM", "LIỂM", "NIỂM", "TRIỂN", "BIỂU", "HIỂU", "TIỂU", "CHIỂU", "THIỂU",
    "Ổ", "BỔ", "CỔ", "ĐỔ", "HỔ", "LỔ", "MỔ", "NỔ", "SỔ", "THỔ", "TỔ",
    "ẨM", "ẨN", "ẨU", "BẨN", "CẨN", "ĐẨY", "HẨM", "KẨU", "LẨN", "MẨN", "TẨY",
    "ẲNG", "BẲNG", "CẲNG", "ĐẲNG", "THẲNG", "LẲNG",
    "Ở", "BỞ", "CỞ", "DỞ", "ĐỞ", "GỞ", "HỞ", "LỞ", "MỞ", "NỞ", "SỞ", "THỞ",
    "CỬA", "LỬA", "NỬA", "SỬA", "THỬA", "GỬI", "THỬ", "CHỬI",
    # Dấu nặng dưới mũ/móc (.): ậ, ặ, ệ, ộ, ợ, ự
    "TIỆM", "DIỆM", "HIỆM", "KIỆM", "LIỆM", "NIỆM", "VIỆM", "TRIỆM", "CHIỆM",
    "ĐIỆN", "HIỆN", "KIỆN", "LIỆN", "NIỆN", "VIỆN", "TRIỆN", "CHIỆN", "TIỆN", "MIỆN",
    "TẬP", "ĐẬP", "CẬP", "HẬP", "LẬP", "MẬP", "GẬP", "BẬP", "THẬP", "NHẬP",
    "MẬT", "THẬT", "CHẬT", "GẬT", "BẬT", "LẬT", "TẬT", "PHẬT", "CẬT", "QUẬT",
    "ĐẬU", "HẬU", "LẬU", "MẬU", "CẬU", "TẬU", "NHẬU", "BẬU",
    "GẶP", "CẶP", "LẶP", "MẶP", "NẶP", "BẶP", "ĐẶC", "MẶC", "NẶNG", "TẶNG",
    "ĐỘI", "TỘI", "CỘI", "HỘI", "BỘI", "LỘI", "NỘI", "RỘI", "VỘI",
    "LỢI", "SỢI", "CHỢ", "THỢ", "TRỢ", "VỢ", "NỢ", "BỢ", "CỢ", "HỢP",
    "LỰC", "MỰC", "CỰC", "ĐỰC", "NỰC", "RỰC", "VỰC", "BỰC", "DỰA", "LỰA", "THỰC"
]

# Âm tiết cơ bản để sinh tổ hợp đa dạng
CONSONANTS_INIT = [
    "", "B", "C", "CH", "D", "Đ", "G", "GH", "H", "K", "KH",
    "L", "M", "N", "NG", "NGH", "NH", "P", "PH", "QU", "R", "S", "T", "TH", "TR", "V", "X"
]
VOWEL_NUCLEI = [
    "A", "Ă", "Â", "E", "Ê", "I", "O", "Ô", "Ơ", "U", "Ư", "Y",
    "AI", "AO", "AU", "AY", "ÂY", "EO", "ÊU", "IA", "IÊ", "IÊU",
    "OA", "OAI", "OAY", "OAN", "OANG", "OE", "OI", "ÔI", "ƠI",
    "UA", "UÔ", "UÔI", "UƠ", "UYA", "UYÊN", "UY", "ƯA", "ƯƠ", "ƯƠI", "ƯƠU"
]
CONSONANTS_END = ["", "C", "CH", "M", "N", "NG", "NH", "P", "T"]

TONE_MAP = {
    'A': ['A', 'Á', 'À', 'Ả', 'Ã', 'Ạ'],
    'Ă': ['Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ'],
    'Â': ['Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ'],
    'E': ['E', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ'],
    'Ê': ['Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ'],
    'I': ['I', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị'],
    'O': ['O', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ'],
    'Ô': ['Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ'],
    'Ơ': ['Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ'],
    'U': ['U', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ'],
    'Ư': ['Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự'],
    'Y': ['Y', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ'],
}

def generate_random_syllable():
    """Sinh ngẫu nhiên một âm tiết tiếng Việt hợp lệ kèm dấu thanh ngẫu nhiên"""
    c_init = random.choice(CONSONANTS_INIT)
    nucleus = random.choice(VOWEL_NUCLEI)
    c_end = random.choice(CONSONANTS_END)
    tone_idx = random.randint(0, 5) # 0: ngang, 1: sắc, 2: huyền, 3: hỏi, 4: ngã, 5: nặng

    # Đặt dấu vào nguyên âm chính
    accented_nucleus = ""
    placed = False
    for char in nucleus:
        if not placed and char in TONE_MAP:
            accented_nucleus += TONE_MAP[char][tone_idx]
            placed = True
        else:
            accented_nucleus += char

    word = f"{c_init}{accented_nucleus}{c_end}"
    return word

def get_random_text(hard_ratio=0.6):
    """Lấy ngẫu nhiên từ vựng tiếng Việt theo tỷ lệ ưu tiên từ khó và biển hiệu"""
    r = random.random()
    if r < hard_ratio * 0.5:
        # Nhóm từ dấu kép khó
        text = random.choice(HARD_DIACRITIC_WORDS)
    elif r < hard_ratio:
        # Nhóm từ biển hiệu thực tế
        text = random.choice(REAL_WORLD_WORDS)
    elif r < 0.90:
        # Nhóm âm tiết ngẫu nhiên
        text = generate_random_syllable()
    else:
        # Số điện thoại, giá tiền, mã số
        choice_type = random.choice(["phone", "price", "code"])
        if choice_type == "phone":
            prefix = random.choice(["0905", "0912", "0988", "0934", "028", "024"])
            num = "".join([str(random.randint(0, 9)) for _ in range(6)])
            text = f"{prefix}.{num[:3]}.{num[3:]}"
        elif choice_type == "price":
            price = random.choice(["25", "30", "35", "40", "50", "100", "150"])
            text = f"{price}.000Đ" if random.random() < 0.5 else f"{price}K"
        else:
            text = f"NO.{random.randint(1, 99)}"

    # Chọn ngẫu nhiên cách viết hoa/thường: 65% In hoa, 25% Viết hoa đầu từ, 10% Chữ thường
    case_r = random.random()
    if case_r < 0.65:
        text = text.upper()
    elif case_r < 0.90:
        text = text.title()
    else:
        text = text.lower()

    return text.strip()


# ==============================================================================
# 2. QUẢN LÝ FONT CHỮ & KIỂM TRA TƯƠNG THÍCH TIẾNG VIỆT
# ==============================================================================

def has_glyph(font, char):
    """Kiểm tra font có hỗ trợ ký tự char hay không (tránh lỗi ô vuông tofu)"""
    try:
        notdef_mask = font.getmask('\ufffe')
        char_mask = font.getmask(char)
        if char_mask.size == (0, 0):
            return False
        if char_mask.size == notdef_mask.size and bytes(char_mask) == bytes(notdef_mask):
            return False
        return True
    except Exception:
        return False

def scan_system_fonts(custom_font_dir=None):
    """Quét và lọc các font có hỗ trợ tiếng Việt trên hệ thống hoặc thư mục chỉ định"""
    search_dirs = []
    if custom_font_dir and os.path.exists(custom_font_dir):
        search_dirs.append(custom_font_dir)

    # Thư mục font trên macOS và Linux
    search_dirs.extend([
        "/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/System/Library/Fonts",
        "/usr/share/fonts",
        "/usr/share/fonts/truetype",
        "./fonts"
    ])

    test_chars = ['đ', 'Đ', 'ễ', 'Ễ', 'ặ', 'Ặ', 'ộ', 'Ộ', 'ử', 'Ử', 'ơ', 'ư']
    valid_fonts = []

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, _, files in os.walk(s_dir):
            for file in files:
                if file.lower().endswith(('.ttf', '.otf')) and not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    try:
                        f = ImageFont.truetype(full_path, 32)
                        # Kiểm tra xem font có chứa đầy đủ các ký tự tiếng Việt cốt lõi không
                        if all(has_glyph(f, ch) for ch in test_chars):
                            valid_fonts.append(full_path)
                    except Exception:
                        pass

    # Loại bỏ trùng lặp
    valid_fonts = list(set(valid_fonts))
    return valid_fonts


# ==============================================================================
# 3. KỸ THUẬT RENDER ẢNH: NỀN, CHỮ, VIỀN, BÓNG & HIỆU ỨNG NHIỄU THỰC TẾ
# ==============================================================================

PALETTES = [
    # Biển hiệu đỏ chữ vàng / trắng
    {"bg": (180, 20, 20), "fg": (255, 230, 0), "stroke": (20, 20, 20)},
    {"bg": (200, 30, 30), "fg": (255, 255, 255), "stroke": (40, 40, 40)},
    # Biển hiệu xanh lá chữ đỏ / trắng (như img01, img02)
    {"bg": (30, 140, 50), "fg": (220, 20, 20), "stroke": (255, 255, 255)},
    {"bg": (40, 160, 60), "fg": (255, 255, 255), "stroke": (0, 80, 20)},
    # Biển hiệu xanh dương chữ vàng / trắng (như img09, img10)
    {"bg": (20, 70, 150), "fg": (255, 220, 20), "stroke": (10, 10, 10)},
    {"bg": (30, 80, 160), "fg": (255, 255, 255), "stroke": (10, 30, 80)},
    # Chữ đèn LED phát sáng trên nền tối (như img03, img04, img05)
    {"bg": (15, 15, 15), "fg": (255, 235, 120), "stroke": (255, 180, 30), "glow": True},
    {"bg": (20, 20, 25), "fg": (255, 255, 255), "stroke": (100, 180, 255), "glow": True},
    # Nền vàng chữ đỏ / đen
    {"bg": (240, 200, 30), "fg": (200, 20, 20), "stroke": (255, 255, 255)},
    {"bg": (245, 215, 40), "fg": (20, 20, 20), "stroke": (255, 255, 255)},
    # Nền trắng chữ đen / đỏ / xanh (biển mica, giấy, nhãn)
    {"bg": (245, 245, 245), "fg": (25, 25, 25), "stroke": None},
    {"bg": (250, 250, 250), "fg": (190, 20, 20), "stroke": None},
]

def make_background(w, h, base_color, bg_style="gradient"):
    """Tạo nền ảnh: màu đơn sắc, gradient chuyển màu hoặc vân tôn sóng"""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    r, g, b = base_color

    if bg_style == "solid":
        arr[:, :, 0] = r
        arr[:, :, 1] = g
        arr[:, :, 2] = b

    elif bg_style == "gradient":
        # Gradient tuyến tính (sáng dần hoặc tối dần)
        factor = random.uniform(0.6, 1.4)
        r2 = min(255, max(0, int(r * factor)))
        g2 = min(255, max(0, int(g * factor)))
        b2 = min(255, max(0, int(b * factor)))
        for y in range(h):
            alpha = y / float(max(1, h))
            arr[y, :, 0] = int(r * (1 - alpha) + r2 * alpha)
            arr[y, :, 1] = int(g * (1 - alpha) + g2 * alpha)
            arr[y, :, 2] = int(b * (1 - alpha) + b2 * alpha)

    elif bg_style == "corrugated":
        # Mô phỏng nền tôn sóng (như trong ảnh thực tế img09.jpeg)
        freq = random.uniform(0.15, 0.35)
        for x in range(w):
            wave = int(25 * np.sin(x * freq))
            arr[:, x, 0] = min(255, max(0, r + wave))
            arr[:, x, 1] = min(255, max(0, g + wave))
            arr[:, x, 2] = min(255, max(0, b + wave))

    # Thêm nhiễu hạt nhẹ ngẫu nhiên để tăng tính thực tế
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_occlusion(img):
    """Mô phỏng vật cản che khuất một phần chữ (như lá cây trong img08.jpeg)"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    num_occlusions = random.randint(1, 3)
    for _ in range(num_occlusions):
        ox = random.randint(int(w * 0.1), int(w * 0.8))
        oy = random.randint(0, h)
        radius_x = random.randint(6, 18)
        radius_y = random.randint(6, 18)
        occ_color = random.choice([
            (30, 80, 30),  # Màu lá cây xanh sẫm
            (20, 20, 20),  # Vật cản đen / cột biển báo
            (180, 180, 180) # Dây điện / vết xước
        ])
        draw.ellipse([ox - radius_x, oy - radius_y, ox + radius_x, oy + radius_y], fill=occ_color)
    return img

def apply_glare_or_flare(img):
    """Mô phỏng quầng lóa sáng (như chóa đèn trong img07.png)"""
    w, h = img.size
    glare_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glare_layer)
    cx = random.randint(0, w)
    cy = random.randint(0, int(h * 0.6))
    radius = random.randint(15, 45)
    glare_color = (255, 255, random.randint(200, 255), random.randint(120, 210))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=glare_color)
    glare_layer = glare_layer.filter(ImageFilter.GaussianBlur(radius=random.randint(6, 12)))
    img = Image.alpha_composite(img.convert("RGBA"), glare_layer).convert("RGB")
    return img


# ==============================================================================
# 4. HÀM SINH MỘT ẢNH MẪU ĐƠN LẺ
# ==============================================================================

def render_single_word(text, font_path, target_height=32):
    """Render một từ tiếng Việt hoàn chỉnh thành ảnh PIL và nhãn ground truth"""
    # 1. Khởi tạo font chữ với size phù hợp
    font_size = random.randint(28, 42)
    font = ImageFont.truetype(font_path, font_size)

    # Đảm bảo 100% font vẽ được tất cả ký tự trong text mà không bị ô vuông
    if not all(has_glyph(font, ch) for ch in text):
        return None

    # 2. Tính kích thước chữ
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return None

    # Padding hai bên và trên dưới
    pad_x = random.randint(10, 22)
    pad_y = random.randint(6, 14)
    canvas_w = tw + pad_x * 2
    canvas_h = th + pad_y * 2

    # 3. Chọn bảng màu & phong cách nền
    palette = random.choice(PALETTES)
    bg_style = random.choice(["solid", "gradient", "gradient", "corrugated"])
    img = make_background(canvas_w, canvas_h, palette["bg"], bg_style=bg_style)
    draw = ImageDraw.Draw(img)

    # 4. Tọa độ chữ chính
    tx = (canvas_w - tw) // 2 - bbox[0]
    ty = (canvas_h - th) // 2 - bbox[1]

    # 5. Vẽ bóng đổ (Drop Shadow) hoặc viền (Stroke)
    use_shadow = random.random() < 0.35
    if use_shadow:
        shadow_offset = random.randint(2, 3)
        draw.text((tx + shadow_offset, ty + shadow_offset), text, font=font, fill=(10, 10, 10))

    use_stroke = palette.get("stroke") is not None and random.random() < 0.70
    if use_stroke:
        stroke_w = random.randint(2, 3)
        draw.text((tx, ty), text, font=font, fill=palette["fg"],
                  stroke_width=stroke_w, stroke_fill=palette["stroke"])
    else:
        draw.text((tx, ty), text, font=font, fill=palette["fg"])

    # 6. Hiệu ứng chữ phát sáng Neon/LED (nếu có glow)
    if palette.get("glow", False):
        glow_layer = img.copy().filter(ImageFilter.GaussianBlur(radius=2))
        img = Image.blend(img, glow_layer, alpha=0.4)

    # 7. Áp dụng các hiệu ứng suy giảm thực tế ngẫu nhiên
    rand_aug = random.random()
    if rand_aug < 0.12:
        # Giả lập vật cản che khuất (như img08.jpeg)
        img = apply_occlusion(img)
    elif rand_aug < 0.22:
        # Giả lập lóa sáng (như img07.png)
        img = apply_glare_or_flare(img)
    elif rand_aug < 0.35:
        # Giả lập nhòe mờ (như img06.png)
        radius = random.uniform(0.6, 1.4)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 8. Xoay góc nhẹ ngẫu nhiên (±4 độ)
    if random.random() < 0.25:
        rot_angle = random.uniform(-4, 4)
        img = img.rotate(rot_angle, resample=Image.BICUBIC, expand=False, fillcolor=palette["bg"])

    # 9. Resize về chiều cao chuẩn định dạng OCR (ví dụ 32px), giữ nguyên tỷ lệ khung hình
    new_w = max(32, int(canvas_w * (target_height / float(canvas_h))))
    img = img.resize((new_w, target_height), Image.LANCZOS)

    return img


# ==============================================================================
# 5. QUY TRÌNH SINH DỮ LIỆU HÀNG LOẠT VÀ XUẤT LABEL CHUẨN OPENOCR
# ==============================================================================

def worker_generate(task_args):
    """Hàm worker xử lý sinh từng ảnh cho đa luồng"""
    sample_idx, output_img_path, fonts, hard_sample_ratio, target_h = task_args
    for _ in range(6):
        try:
            text = get_random_text(hard_ratio=hard_sample_ratio)
            chosen_font = random.choice(fonts)
            img = render_single_word(text, chosen_font, target_height=target_h)
            if img is not None:
                # Lưu ảnh với chất lượng nén JPEG thực tế (70 - 95)
                quality = random.randint(70, 95)
                img.save(output_img_path, "JPEG", quality=quality)
                return output_img_path, text
        except Exception:
            pass
    return None, None


def generate_dataset(output_dir, num_samples, val_ratio, font_dir, hard_sample_ratio, target_h, workers):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"🔍 Đang tìm kiếm font chữ tiếng Việt...")
    fonts = scan_system_fonts(font_dir)
    if len(fonts) == 0:
        print("❌ Không tìm thấy font tiếng Việt nào! Vui lòng cài đặt font hoặc cung cấp thư mục --font_dir.")
        sys.exit(1)
    print(f"✅ Tìm thấy {len(fonts)} font chữ hỗ trợ đầy đủ tiếng Việt.")

    print(f"🚀 Bắt đầu sinh {num_samples} ảnh tiếng Việt tổng hợp...")
    tasks = []
    for i in range(num_samples):
        img_name = f"word_{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        tasks.append((i, img_path, fonts, hard_sample_ratio, target_h))

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res_path, res_text in tqdm(executor.map(worker_generate, tasks), total=num_samples, desc="Tạo sinh"):
            if res_path and res_text:
                rel_path = os.path.relpath(res_path, output_dir)
                results.append((rel_path, res_text))

    print(f"✅ Đã tạo thành công {len(results)}/{num_samples} ảnh.")

    # Phân chia tập Train và Validation
    random.shuffle(results)
    num_val = int(len(results) * val_ratio)
    val_set = results[:num_val]
    train_set = results[num_val:]

    train_gt_path = os.path.join(output_dir, "rec_gt_train.txt")
    val_gt_path = os.path.join(output_dir, "rec_gt_val.txt")

    # Ghi file label theo định dạng OpenOCR: relative_path \t label
    with open(train_gt_path, "w", encoding="utf-8") as f:
        for rel_path, text in train_set:
            f.write(f"{rel_path}\t{text}\n")

    with open(val_gt_path, "w", encoding="utf-8") as f:
        for rel_path, text in val_set:
            f.write(f"{rel_path}\t{text}\n")

    print("\n" + "=" * 60)
    print("🎉 HOÀN TẤT SINH DỮ LIỆU!")
    print(f"📁 Thư mục ảnh:      {images_dir}")
    print(f"📄 Label Train:      {train_gt_path} ({len(train_set)} mẫu)")
    print(f"📄 Label Validation: {val_gt_path} ({len(val_set)} mẫu)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Synthetic Text Generator for OpenOCR / SVTRv2")
    parser.add_argument("--output_dir", type=str, default="./synth_vn_data", help="Thư mục xuất dữ liệu")
    parser.add_argument("--num_samples", type=int, default=10000, help="Số lượng mẫu cần sinh")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỷ lệ tập validation")
    parser.add_argument("--font_dir", type=str, default="./fonts", help="Thư mục font tùy chỉnh")
    parser.add_argument("--hard_sample_ratio", type=float, default=0.65, help="Tỷ lệ ưu tiên từ dấu kép & biển hiệu")
    parser.add_argument("--img_height", type=int, default=32, help="Chiều cao ảnh chuẩn hóa (mặc định 32px)")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng xử lý song song")

    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        val_ratio=args.val_ratio,
        font_dir=args.font_dir,
        hard_sample_ratio=args.hard_sample_ratio,
        target_h=args.img_height,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
