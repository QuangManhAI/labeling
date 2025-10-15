# %%
import torch, cv2
from utils.util import non_max_suppression, non_max_suppression_editable, load_thresholds_from_args_yaml

import yaml

# %%
# ---------- params ----------
WEIGHT = "internal_assets/weights/best.pt"
IMG_PATH = ["internal_assets/dataset/VietNam_street.png",
            "internal_assets/dataset/Highway.png",
            "internal_assets/dataset/VN_Highway.png",
            "internal_assets/dataset/Result_image.jpg"]
DATA_YAML = "utils/args.yaml"   # Path to dataset yaml
INPUT_SIZE = (640, 640) 
CONF_THR = 0.25
IOU_THR = 0.45


# %%
with open(DATA_YAML, "r") as f:
    data_dict = yaml.safe_load(f)

names = data_dict["names"]   # dict {0:"person",1:"bicycle",...}


# %%
# ---------- utils ----------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # img: BGR numpy (H, W, C) as loaded by cv2
    h0, w0 = img.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h0, new_w / w0)
    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))
    # resize
    img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    # compute padding
    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))
    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)


# %%
# ---------- load model ----------
ckpt = torch.load(WEIGHT, map_location="cpu", weights_only=False)
if 'model' in ckpt:
    model = ckpt['model']
else:
    raise RuntimeError("Checkpoint does not contain 'model' key.")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# prefer float for numeric stability
if next(model.parameters()).dtype == torch.half:
    model = model.half()
else:
    model = model.float()


# %%
# Assume you already have: model, device, names, letterbox() from before

def load_images_as_batch(img_paths, new_shape=(640, 640)):
    """
    Loads multiple images, applies letterbox resize, and returns:
    - batch tensor [B, 3, H, W]
    - list of (original_h, original_w)
    - list of (gain, (pad_w, pad_h))
    """
    imgs_tensor = []
    shapes = []
    ratios = []

    for path in img_paths:
        img_bgr = cv2.imread(path) #np.array (H, W, 3)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {path}")

        orig_h, orig_w = img_bgr.shape[:2] #np.array (H, W)
        img_pad, gain, (pad_w, pad_h) = letterbox(img_bgr, new_shape=new_shape)
        '''
Gọi hàm letterbox để resize ảnh về kích thước chuẩn của YOLO (INPUT_SIZE, ví dụ 640×640).
Letterbox = resize ảnh nhưng vẫn giữ tỉ lệ khung hình (aspect ratio) → phần thừa sẽ được padding màu đen.
Trả về:
img_pad: ảnh sau khi resize + pad.
gain: hệ số scale (ảnh gốc → ảnh mới).
(pad_w, pad_h): độ pad thêm ở 2 chiều.
Thông tin gain, pad_w, pad_h được dùng sau này để chuyển ngược bbox từ ảnh YOLO về ảnh gốc.
'''
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB) #Chuyển từ BGR → RGB (YOLO và PyTorch thường chuẩn hóa input thành RGB).
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        '''
torch.from_numpy(img_rgb): chuyển ảnh NumPy → Tensor PyTorch.
.permute(2, 0, 1): đổi trục từ (H, W, C) → (C, H, W) (PyTorch format).
.unsqueeze(0): thêm batch dimension → [1, C, H, W].
.float() / 255.0: đổi từ uint8 (0–255) sang float32 (0–1) để mạng dễ học
'''
        imgs_tensor.append(tensor)
        shapes.append((orig_h, orig_w))
        ratios.append((gain, pad_w, pad_h))
        

    batch_tensor = torch.stack(imgs_tensor, 0)  # [B,3,H,W]
    return batch_tensor, shapes, ratios


def run_inference_batch(model, img_paths, conf_thres=0.25, iou_thres=0.45):
    batch_tensor, shapes, ratios = load_images_as_batch(img_paths, new_shape=(640,640))
    batch_tensor = batch_tensor.to(device)

    if next(model.parameters()).dtype == torch.half:
        batch_tensor = batch_tensor.half()
        '''
    Nếu mô hình đang ở dạng half precision (FP16) → convert input sang .half() để đồng bộ.
    Dùng khi inference trên GPU để tăng tốc và tiết kiệm bộ nhớ.
'''

    with torch.no_grad():
        preds = model(batch_tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

    # detections = non_max_suppression(preds, confidence_threshold=conf_thres, iou_threshold=iou_thres)
    
    per_class_dict, per_class_list, names = load_thresholds_from_args_yaml('utils/args.yaml', default_threshold=CONF_THR)

    # Then call your NMS:
    # Option A: pass dict (works with the NMS implementation I gave earlier)
    detections = non_max_suppression_editable(preds, confidence_threshold=per_class_dict, iou_threshold=0.45, default_threshold=CONF_THR)

    results = []
    for i, det in enumerate(detections):
        orig_h, orig_w = shapes[i]
        gain, pad_w, pad_h = ratios[i]

        if det is not None and len(det):
            det = det.clone()
            # de-letterbox
            det[:, [0, 2]] -= pad_w
            det[:, [1, 3]] -= pad_h
            det[:, :4] /= gain
            det[:, [0, 2]] = det[:, [0, 2]].clamp(0, orig_w)
            det[:, [1, 3]] = det[:, [1, 3]].clamp(0, orig_h)
        results.append(det)
    return results

        # BGR colors
CLASS_COLORS = {
    "person": (0, 0, 255),        # red
    "motorcycle": (0, 255, 255),  # yellow
    "car": (255, 255, 255),       # white
}


# %%
detections = run_inference_batch(model, IMG_PATH)

for path, det in zip(IMG_PATH, detections):
    img = cv2.imread(path)
    if det is not None:
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            color = CLASS_COLORS.get(cls_name, (0,125,0))
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, max(y1-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    save_path = f"internal_assets/inference_result/inference_result_{path.split('/')[-1]}"
    cv2.imwrite(save_path, img)
    print(f"Saved {save_path}")



