import cv2
import os
from pathlib import Path

def concatenate_videos(video_files, output_name="combined_video.mp4"):
    if not video_files:
        print("沒有找到影片檔案！")
        return

    # 讀取第一部影片獲取參數 (FPS, 寬, 高)
    cap = cv2.VideoCapture(str(video_files[0]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 設定影片編碼器 (mp4v 是常用且相容性高的編碼)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    print(f"基準參數: {width}x{height} @ {fps} FPS")
    print(f"開始串連 {len(video_files)} 部影片...")

    for i, video_path in enumerate(video_files):
        print(f"正在處理 ({i+1}/{len(video_files)}): {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 如果後續影片尺寸不同，強制縮放到第一部的尺寸以免出錯
            if (frame.shape[1] != width) or (frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
        
        cap.release()

    out.release()
    print(f"✅ 串連完成！儲存為: {output_name}")

# ==================== 使用範例 ====================
if __name__ == "__main__":
    # 方式 A: 手動列出檔案順序 (推薦，確保 Episode 0->1->2->3 順序正確)
    # videos_to_combine = [
    #     "./videos/combined_all_poses_center_semantic.mp4",
    #     "./videos/combined_all_poses_up_semantic.mp4",
    #     "./videos/combined_all_poses_down_semantic.mp4",
    #     "./videos/combined_all_poses_left_semantic.mp4",
    #     "./videos/combined_all_poses_right_semantic.mp4"
    # ]
    videos_to_combine = [
        "./videos/ep0_cube2_pose0_up_wrist_semantic.mp4",
        "./videos/ep0_cube2_pose0_up_wrist_semantic.mp4",
        "./videos/ep0_cube2_pose0_up_wrist_semantic.mp4",
        "./videos/ep0_cube2_pose0_up_wrist_semantic.mp4",
        "./videos/ep0_cube2_pose0_up_wrist_semantic.mp4",
        "./videos/ep1_cube3_pose1_up_wrist_semantic.mp4",
        "./videos/ep1_cube3_pose1_up_wrist_semantic.mp4",
        "./videos/ep1_cube3_pose1_up_wrist_semantic.mp4",
        "./videos/ep1_cube3_pose1_up_wrist_semantic.mp4",
        "./videos/ep1_cube3_pose1_up_wrist_semantic.mp4",
        "./videos/ep2_cube4_pose2_up_wrist_semantic.mp4",
        "./videos/ep2_cube4_pose2_up_wrist_semantic.mp4",
        "./videos/ep2_cube4_pose2_up_wrist_semantic.mp4",
        "./videos/ep2_cube4_pose2_up_wrist_semantic.mp4",
        "./videos/ep2_cube4_pose2_up_wrist_semantic.mp4",
        "./videos/ep3_cube5_pose3_up_wrist_semantic.mp4",
        "./videos/ep3_cube5_pose3_up_wrist_semantic.mp4",
        "./videos/ep3_cube5_pose3_up_wrist_semantic.mp4",
        "./videos/ep3_cube5_pose3_up_wrist_semantic.mp4",
        "./videos/ep3_cube5_pose3_up_wrist_semantic.mp4",
    ]
    
    # 方式 B: 自動抓取目錄下所有 mp4 (注意排序可能不符預期)
    # video_folder = Path("./videos")
    # videos_to_combine = sorted(list(video_folder.glob("*.mp4")))

    concatenate_videos(videos_to_combine, "./videos/file-000.mp4")