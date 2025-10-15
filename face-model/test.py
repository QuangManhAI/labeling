from pymongo import MongoClient
import cv2
import os

# --- 1️⃣ Kết nối MongoDB ---
uri = "mongodb+srv://manh:200406@npqm.5lvoaxo.mongodb.net/"
client = MongoClient(uri)
db = client["face_ml_labeling"]      # ⚠️ đổi theo tên DB của bạn
collection = db["images"]         # ⚠️ đổi theo tên collection bạn dùng

# --- 2️⃣ Cấu hình ---
BASE_DIR = "/home/quangmanh/Documents/project_1/face-ml-labeling"  # gốc thư mục ảnh
COLORS = {"truck": (0, 0, 255), "car": (0, 255, 255), "person": (255, 255, 255)}

# --- 3️⃣ Lặp qua từng ảnh có annotations ---
for doc in collection.find({"annotations": {"$exists": True, "$ne": []}}):
    file_path = os.path.join(BASE_DIR, doc["filePath"].lstrip("/"))
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy ảnh: {file_path}")
        continue

    img = cv2.imread(file_path)
    if img is None:
        print(f"❌ Lỗi đọc ảnh: {file_path}")
        continue

    # --- 4️⃣ Vẽ bounding boxes ---
    for ann in doc["annotations"]:
        x1, y1, x2, y2 = map(int, ann["bbox"])
        label = ann.get("label", "unknown")
        conf = ann.get("confidence", 0.0)
        color = COLORS.get(label, (0, 255, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # --- 5️⃣ Lưu ảnh có box ---
    name, ext = os.path.splitext(file_path)
    out_path = f"{name}_1{ext}"
    cv2.imwrite(out_path, img)
    print(f"✅ Đã lưu: {out_path}")

client.close()
