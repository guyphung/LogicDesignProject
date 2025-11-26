import torch
import cv2
import easyocr
import pathlib
import sys
import numpy as np
import os

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(CURRENT_DIR / 'best_30.pt')

if not os.path.exists(MODEL_PATH):
    print(f"Không tìm thấy file model")
    sys.exit()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

print(f"✅ Đã tìm thấy model tại: {MODEL_PATH}")
CONF_THRESH = 0.5
SKIP_FRAMES = 15

print("-> Đang tải model YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

print("-> Đang khởi động EasyOCR...")
reader = easyocr.Reader(['en']) 

# --- 4. LOGIC SỬA LỖI ---
dict_char_to_num = {'O': '0', 'D': '0', 'Q': '0', 'U': '0', 'I': '1', 'L': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'H': '8', 'g': '9', 'q': '9'}
dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}

def xu_ly_hau_ky(text):
    text = text.upper()
    text_list = list(text)
    
    text_list = [c for c in text_list if c.isalnum()]
    if len(text_list) < 3: return "".join(text_list)

    for i in range(min(2, len(text_list))):
        if text_list[i] in dict_char_to_num:
            text_list[i] = dict_char_to_num[text_list[i]]

    if len(text_list) > 2:
        if text_list[2] in dict_num_to_char:
             text_list[2] = dict_num_to_char[text_list[2]]
    
    for i in range(4, len(text_list)): 
        if text_list[i] in dict_char_to_num:
             text_list[i] = dict_char_to_num[text_list[i]]

    return "".join(text_list)

def doc_bien_so_thong_minh(img_crop):
    img_crop = cv2.resize(img_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_crop = cv2.copyMakeBorder(img_crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    result = reader.readtext(gray, detail=0)
    text = "".join(result)
    text = ''.join(e for e in text if e.isalnum())
    
    return xu_ly_hau_ky(text)

print("-> Opening Camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

frame_count = 0
current_text = ""   
is_square_plate = False 

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    results = model(frame)
    detections = results.xyxy[0].numpy()
    
    plate_found = False

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
        if conf > CONF_THRESH:
            plate_found = True
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            h_plate = y2 - y1
            w_plate = x2 - x1
            if h_plate / w_plate > 0.5:
                is_square_plate = True
            else:
                is_square_plate = False

            if frame_count % SKIP_FRAMES == 0:
                try:
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size > 0:
                        text_read = doc_bien_so_thong_minh(plate_crop)
                        if len(text_read) > 3:
                            current_text = text_read
                            print(f"Biển số: {current_text}")
                except: pass

            if current_text:
                if is_square_plate and len(current_text) >= 7:
                    split_index = 4 if len(current_text) >= 8 else 3
                    
                    line1 = current_text[:split_index]
                    line2 = current_text[split_index:]
                    
                    cv2.rectangle(frame, (x1, y1-60), (x2, y1), (0, 0, 0), -1)
                    cv2.putText(frame, line1, (x1 + 5, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, line2, (x1 + 5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    # Biển dài
                    cv2.rectangle(frame, (x1, y1-40), (x2, y1), (0, 0, 0), -1)
                    cv2.putText(frame, current_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if not plate_found and frame_count % 60 == 0:
        current_text = ""

    cv2.imshow('CAMERA (Press Q to Quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pathlib.PosixPath = temp