import cv2
import os
from pathlib import Path

def split_video_by_reference(ref_a_path, ref_b_path, target_c_path):
    # --- 檔案存在性檢查 ---
    paths = [ref_a_path, ref_b_path, target_c_path]
    names = ["參考 A", "參考 B", "目標 C"]
    
    for name, p in zip(names, paths):
        p_obj = Path(p)
        exists = p_obj.exists()
        size = p_obj.stat().st_size if exists else 0
        print(f"[{'OK' if exists else 'ERR'}] {name}: {p} (大小: {size / 1024 / 1024:.2f} MB)")
        if not exists:
            print(f"❌ 找不到檔案: {p}")
            return

    # 1. 獲取參考影片 A 和 B 的長度
    def get_len(path, name):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"❌ OpenCV 無法開啟 {name}，請檢查路徑是否包含特殊字元或格式是否支援。")
            return 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length

    len_a = get_len(ref_a_path, "影片 A")
    len_b = get_len(ref_b_path, "影片 B")
    
    if len_a == 0 or len_b == 0:
        print("🛑 偵測到 Frame 數為 0，停止操作。請確認是否安裝了 FFMPEG 解碼器。")
        return
    
    print(f"✅ 參考長度確定: A = {len_a} 幀, B = {len_b} 幀")

    # 2. 準備處理目標影片 C
    cap_c = cv2.VideoCapture(str(target_c_path))
    if not cap_c.isOpened():
        print("❌ 無法開啟目標影片 C 進行切割。")
        return

    fps = cap_c.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # 防呆
    
    width = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 輸出路徑 (建議存在目前目錄避免權限問題)
    out_c1_path = "/home/mike/.cache/huggingface/lerobot/MikeChenYZ/soarm-fmb-x5/videos/observation.images.top/chunk-000/chunk-000_new.mp4"
    out_c2_path = "/home/mike/.cache/huggingface/lerobot/MikeChenYZ/soarm-fmb-x5/videos/observation.images.top/chunk-000/chunk-001_new.mp4"

    # 3. 切割 Part 1
    writer1 = cv2.VideoWriter(out_c1_path, fourcc, fps, (width, height))
    print(f"🚀 正在切割 Part 1 ({len_a} 幀)...")
    for i in range(len_a):
        ret, frame = cap_c.read()
        if not ret:
            print(f"⚠️ Part 1 在第 {i} 幀提前中斷")
            break
        writer1.write(frame)
    writer1.release()

    # 4. 切割 Part 2
    writer2 = cv2.VideoWriter(out_c2_path, fourcc, fps, (width, height))
    print(f"🚀 正在切割 Part 2 ({len_b} 幀)...")
    for i in range(len_b):
        ret, frame = cap_c.read()
        if not ret:
            print(f"⚠️ Part 2 在第 {i} 幀提前中斷")
            break
        writer2.write(frame)
    writer2.release()

    cap_c.release()
    print(f"✨ 切割完成！\n儲存為: {out_c1_path} (對齊 A) 與 {out_c2_path} (對齊 B)")

if __name__ == "__main__":
    # 使用 Path 處理路徑較安全
    base_folder = "/home/mike/.cache/huggingface/lerobot/MikeChenYZ/soarm-fmb-x5/videos/observation.images.top/chunk-000"
    
    a = os.path.join(base_folder, "file-000.mp4")
    b = os.path.join(base_folder, "file-001.mp4")
    c = os.path.join(base_folder, "file-002.mp4")
    
    split_video_by_reference(a, b, c)