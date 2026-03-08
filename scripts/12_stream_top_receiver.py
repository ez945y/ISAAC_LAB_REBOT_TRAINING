#!/usr/bin/env python
"""
接收並顯示 stream_top_sender.py 送出的串流影像 (TCP receiver)。

此腳本不需要 LeRobot 或 Isaac Lab 的任何依賴，只需要 cv2 和 numpy。

Sender 會循環回放模擬，每輪都是重新跑物理引擎，
所以在 sender 端修改相機參數後，下一輪就會反映出來。

按鍵:
  q / ESC  — 退出

Usage:
    python scripts/12_stream_top_receiver.py
    python scripts/12_stream_top_receiver.py --port 9999
    python scripts/12_stream_top_receiver.py --host 192.168.1.100 --port 9999
"""

import argparse
import socket
import struct

import cv2
import numpy as np


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """確保接收到指定大小的資料"""
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("連線已關閉")
        buf += chunk
    return buf


def main():
    parser = argparse.ArgumentParser(description="TCP receiver: 接收並顯示串流影像")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Sender 的 IP 位址")
    parser.add_argument("--port", type=int, default=9999,
                        help="TCP port")
    parser.add_argument("--window_name", type=str, default="LeRobot Top Camera",
                        help="視窗名稱")
    args = parser.parse_args()

    print(f"[INFO] 正在連線至 {args.host}:{args.port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[INFO] 已連線！接收串流中... (按 q 退出)")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    try:
        while True:
            # 讀取 4 bytes header (frame size)
            header = recv_exact(sock, 4)
            frame_size = struct.unpack('>I', header)[0]

            # frame_size == 0 表示結束 (不應再出現，但保留相容)
            if frame_size == 0:
                print(f"[INFO] 收到結束信號 (共 {frame_count} 幀)")
                break

            # 讀取 JPEG data
            jpeg_data = recv_exact(sock, frame_size)

            # 解碼 JPEG
            img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 加上幀資訊
            info_text = f"Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(args.window_name, frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print(f"[INFO] 使用者中斷，已接收 {frame_count} 幀")
                break

    except ConnectionError as e:
        print(f"[INFO] 連線結束: {e} (已接收 {frame_count} 幀)")
    except KeyboardInterrupt:
        print(f"\n[INFO] 使用者中斷 (已接收 {frame_count} 幀)")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[INFO] 完成")


if __name__ == "__main__":
    main()
