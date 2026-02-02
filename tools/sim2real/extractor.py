import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Callable, Optional
from transformers import AutoModel
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

@dataclass
class FeatureExtractorConfig:
    model_name: str = "theia-tiny-patch16-224-cddsv"  # or "mobilenet_v3_large", "mobilenet_v3_small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim: int = 192  # Theia-tiny hidden dim, or MobileNet pool dim
    use_fp16: bool = True
    pool_output: bool = False  # 是否自動 mean pool 成單一向量 (推薦)

class FeatureExtractor:
    """獨立版本的特徵提取器，不依賴 Isaac Lab，可用於實體相機。

    支援模型：
    - Theia 系列（預設）
    - MobileNetV3-Large / Small（透過 model_name 切換）

    輸入：torch.Tensor (N, H, W, C) uint8 [0-255] RGB
    輸出：(N, num_patches, hidden_dim) 或 (N, embedding_dim) 如果 pool_output=True
    """

    def __init__(self, cfg: FeatureExtractorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = None
        self.inference_fn: Callable = None

        # 支援的模型類型
        self.supported_models = {
            "theia": self._load_theia,
            "mobilenet_v3_large": self._load_mobilenet_v3_large,
            "mobilenet_v3_small": self._load_mobilenet_v3_small,
        }

        # 根據 model_name 載入
        self._load_model(cfg.model_name)

        print(f"[INFO] Loaded {cfg.model_name} on {cfg.device} (FP16={cfg.use_fp16}, pool={cfg.pool_output})")

    def _load_theia(self):
        """載入 Theia 模型（官方邏輯）"""
        model = AutoModel.from_pretrained(
            f"theaiinstitute/{self.cfg.model_name}",
            trust_remote_code=True
        ).eval().to(self.device)

        def inference(images: torch.Tensor) -> torch.Tensor:
            image_proc = images.to(self.device).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            if self.cfg.use_fp16:
                image_proc = image_proc.half()

            with torch.no_grad():
                features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
                patch_tokens = features.last_hidden_state[:, 1:]

            if self.cfg.pool_output:
                return patch_tokens.mean(dim=1)  # (N, hidden_dim)
            return patch_tokens

        self.model = model
        self.inference_fn = inference

    def _load_mobilenet_v3_large(self):
        """載入 MobileNet V3 Large"""
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights).features.eval().to(self.device)

        def inference(images: torch.Tensor) -> torch.Tensor:
            # Theia 風格 preprocess
            image_proc = images.to(self.device).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            if self.cfg.use_fp16:
                image_proc = image_proc.half()

            with torch.no_grad():
                features = model(image_proc)  # (N, 960, H/32, W/32)

            if self.cfg.pool_output:
                pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(start_dim=1)  # (N, 960)
                return pooled
            return features  # 如果不 pool，返回 feature map

        self.model = model
        self.inference_fn = inference

    def _load_mobilenet_v3_small(self):
        """載入 MobileNet V3 Small（更輕量）"""
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights).features.eval().to(self.device)

        def inference(images: torch.Tensor) -> torch.Tensor:
            image_proc = images.to(self.device).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            if self.cfg.use_fp16:
                image_proc = image_proc.half()

            with torch.no_grad():
                features = model(image_proc)

            if self.cfg.pool_output:
                pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(start_dim=1)  # (N, 576)
                return pooled
            return features

        self.model = model
        self.inference_fn = inference

    def _load_model(self, model_name: str):
        if model_name.startswith("theia"):
            self._load_theia()
        elif model_name == "mobilenet_v3_large":
            self._load_mobilenet_v3_large()
        elif model_name == "mobilenet_v3_small":
            self._load_mobilenet_v3_small()
        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported: theia-*, mobilenet_v3_large/small")

    def step(self, rgb_img: torch.Tensor) -> torch.Tensor:
        """主提取函式。
        
        Args:
            rgb_img: (N, H, W, C) uint8 [0-255] RGB tensor (可從 OpenCV 直接轉)
        
        Returns:
            torch.Tensor: 特徵向量 (N, dim)
        """
        with torch.no_grad():
            features = self.inference_fn(rgb_img)
        return features.clone().detach()

    def __call__(self, rgb_img: torch.Tensor) -> torch.Tensor:
        return self.step(rgb_img)

# 使用範例
if __name__ == "__main__":
    cfg = FeatureExtractorConfig(
        model_name="theia-tiny-patch16-224-cddsv",
        device="cuda",
        pool_output=True
    )
    extractor = FeatureExtractor(cfg)

    import numpy as np
    import cv2
    frame = cv2.imread("test.jpg")  # BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(cfg.device)  # (1, H, W, C)

    emb = extractor(frame_tensor)
    print(emb.shape)  # 如果 pool_output=True → (1, 960) 或 (1, 192)