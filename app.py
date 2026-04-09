# app.py
# 실행: streamlit run app.py
# 패키지: streamlit torch torchvision opencv-python-headless Pillow numpy

import io
import ssl
import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw

# ────────────────────────────────────────────────────────────
# 0. 페이지 설정
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="당뇨병성 망막병증 진단",
    page_icon="👁️",
    layout="wide",
)

# ────────────────────────────────────────────────────────────
# 1. 상수 / 레이블 정의
# ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INFO = {
    0: {"label": "Grade 0 — 정상 (No DR)",          "color": "#2ecc71", "emoji": "✅"},
    1: {"label": "Grade 1 — 경증 (Mild DR)",         "color": "#f1c40f", "emoji": "⚠️"},
    2: {"label": "Grade 2 — 중등도 (Moderate DR)",   "color": "#e67e22", "emoji": "🟠"},
    3: {"label": "Grade 3 — 중증 (Severe DR)",       "color": "#e74c3c", "emoji": "🔴"},
    4: {"label": "Grade 4 — 증식성 (Proliferative)", "color": "#8e44ad", "emoji": "🟣"},
}

MODEL_PATH = "best_model_aptos_weighted.pt"

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ────────────────────────────────────────────────────────────
# 2. 전처리 함수
# ────────────────────────────────────────────────────────────
def preprocess_eye_image(pil_img: Image.Image, img_size: int = 300) -> np.ndarray:
    image = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 10
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if rows.any() and cols.any():
        image = image[np.ix_(rows, cols)]
    h, w, _ = image.shape
    if h > w:
        pad = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        pad = (w - h) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = cv2.resize(image, (img_size, img_size))
    return image


# ────────────────────────────────────────────────────────────
# 3. 모델 로드 (ResNet-50 전용)
# ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 가중치 로딩 중…")
def load_model(model_path: str):
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        if os.path.exists(model_path):
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 5)
            state = torch.load(model_path, map_location=DEVICE)
            m.load_state_dict(state)
            st.sidebar.success("✅ 학습된 가중치 로드 완료")
        else:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            m.fc = nn.Linear(m.fc.in_features, 5)
            st.sidebar.warning(
                f"⚠️ '{model_path}' 를 찾을 수 없습니다.\n"
                "ImageNet pretrained 가중치로 데모 실행 중입니다."
            )
    finally:
        ssl._create_default_https_context = _orig_ctx
    m.to(DEVICE).eval()
    return m


# ────────────────────────────────────────────────────────────
# 4. Grad-CAM
# ────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        # hook을 handle로 저장 → generate() 후 반드시 제거해 누적 방지
        self._hook_handles = [
            target_layer.register_full_backward_hook(self._save_gradient),
            target_layer.register_forward_hook(self._save_activation),
        ]

    def _save_gradient(self, module, grad_input, grad_output):
        for g in grad_output:
            if g is not None:
                self.gradients = g.detach()
                break

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def generate(self, input_tensor, target_class=None):
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()  # .item()으로 안전한 GPU→CPU 변환
            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = F.relu(torch.sum(weights * self.activations, dim=1).squeeze())
            cam = cam.cpu().numpy()
            denom = cam.max() - cam.min()
            cam = (cam - cam.min()) / denom if denom > 1e-8 else np.zeros_like(cam)
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            cam = cv2.resize(cam, (w, h))
            return cam, output
        finally:
            self.remove_hooks()  # 에러 발생 여부와 무관하게 항상 hook 제거


# ────────────────────────────────────────────────────────────
# 5. 시각화 헬퍼
# ────────────────────────────────────────────────────────────
def make_gradcam_image(rgb_img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """원본 + Grad-CAM 히트맵 오버레이"""
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_img, 0.55, heatmap_rgb, 0.45, 0)


def make_bbox_only_image(rgb_img: np.ndarray, cam: np.ndarray, threshold: float = 0.70) -> np.ndarray:
    """원본 이미지에 Bounding Box만 표시 (히트맵 합성 없음)"""
    h, w = rgb_img.shape[:2]
    result = rgb_img.copy()
    thresh_val = int(255 * threshold)
    _, thresh = cv2.threshold(np.uint8(255 * cam), thresh_val, 255, cv2.THRESH_BINARY)
    thresh = cv2.resize(thresh, (w, h))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(result, "Lesion Suspect",
                    (x, max(y - 10, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return result


def make_download_image(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray,
                        titles: list, suptitle: str) -> bytes:
    """PIL로 3장을 가로로 이어붙여 PNG bytes 반환 (matplotlib 불필요)"""
    H, W    = img1.shape[:2]
    TITLE_H = 30
    SUPER_H = 40
    PAD     = 10

    total_w = W * 3 + PAD * 4
    total_h = H + TITLE_H + SUPER_H + PAD * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    draw.text((total_w // 2, SUPER_H // 2), suptitle,
              fill=(255, 255, 255), anchor="mm")

    for i, (arr, title) in enumerate(zip([img1, img2, img3], titles)):
        x_off = PAD + i * (W + PAD)
        y_off = SUPER_H + TITLE_H + PAD
        draw.text((x_off + W // 2, SUPER_H + TITLE_H // 2), title,
                  fill=(200, 200, 200), anchor="mm")
        canvas.paste(Image.fromarray(arr.astype(np.uint8)), (x_off, y_off))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ────────────────────────────────────────────────────────────
# 6. UI
# ────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.image(
            "C:/workspace/dl_project/ex_image.png",
            caption="안저(Fundus) 이미지 예시",
            width=160,
        )
        st.markdown("### ⚙️ 설정")
        model_path_input = st.text_input("가중치 파일 경로", value=MODEL_PATH)
        bbox_threshold = st.slider(
            "Bounding Box 임계값",
            min_value=0.50, max_value=0.95, value=0.70, step=0.05,
            help="Grad-CAM에서 병변으로 판정할 최소 활성화 비율",
        )
        st.markdown("---")
        st.markdown(
            "**진단 등급 안내**\n"
            "- Grade 0: 정상\n- Grade 1: 경증\n- Grade 2: 중등도\n"
            "- Grade 3: 중증\n- Grade 4: 증식성\n"
        )
        st.caption("⚠️ 본 서비스는 연구·교육 목적이며, 실제 의료 진단을 대체하지 않습니다.")

    st.title("👁️ 당뇨병성 망막병증 (DR) 진단 시스템")
    st.markdown(
        "안저(Fundus) 이미지를 업로드하면 **ResNet-50 모델**이 당뇨병성 망막병증 등급을 예측하고,  \n"
        "**Grad-CAM**으로 모델이 주목한 병변 영역을 시각화합니다."
    )
    st.markdown("---")

    model = load_model(model_path_input)

    uploaded = st.file_uploader(
        "🔼 안저 이미지를 업로드하세요 (JPG / PNG / BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    if uploaded is None:
        st.info("이미지를 업로드하면 즉시 분석을 시작합니다.")
        return

    pil_img  = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    proc_rgb = preprocess_eye_image(pil_img)
    tensor   = val_transform(Image.fromarray(proc_rgb)).unsqueeze(0).to(DEVICE)

    with st.spinner("🔍 모델이 이미지를 분석 중입니다…"):
        grad_cam = GradCAM(model, model.layer4[-1])
        with torch.enable_grad():
            cam, output = grad_cam.generate(tensor, target_class=None)
        probs      = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()
        pred_class = int(np.argmax(probs))

    info = CLASS_INFO[pred_class]
    st.markdown("---")
    st.markdown("## 📊 예측 결과")

    col_res, col_conf = st.columns([1, 2])
    with col_res:
        st.markdown(
            f"""<div style="background:{info['color']}22;border-left:6px solid {info['color']};
            border-radius:8px;padding:18px 20px;font-size:1.3rem;font-weight:700;">
            {info['emoji']} &nbsp; {info['label']}</div>""",
            unsafe_allow_html=True,
        )
    with col_conf:
        st.markdown("**각 등급별 예측 확률**")
        for grade, prob in enumerate(probs):
            g_info = CLASS_INFO[grade]
            pct = prob * 100
            st.markdown(
                f"""<div style="margin-bottom:4px;">
                <span style="font-size:0.85rem;color:#555;">{g_info['emoji']} Grade {grade}</span>
                <div style="background:#eee;border-radius:4px;height:18px;width:100%;overflow:hidden;">
                <div style="background:{g_info['color']};width:{pct:.1f}%;height:100%;border-radius:4px;
                display:flex;align-items:center;padding-left:6px;color:white;font-size:0.75rem;
                font-weight:600;">{pct:.1f}%</div></div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## 🖼️ 시각화")
    st.caption("왼쪽부터: (1) 전처리 원본  ·  (2) Grad-CAM 히트맵  ·  (3) 원본 + Bounding Box")

    img_original = proc_rgb
    img_gradcam  = make_gradcam_image(proc_rgb, cam)
    img_bbox     = make_bbox_only_image(proc_rgb, cam, bbox_threshold)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_original, caption="① 원본 (전처리 후)", width="stretch")
    with col2:
        st.image(img_gradcam, caption="② Grad-CAM 히트맵", width="stretch")
    with col3:
        st.image(img_bbox, caption=f"③ 원본 + Bounding Box (임계값 {bbox_threshold:.0%})", width="stretch")

    with st.expander("📋 상세 확률 테이블 보기"):
        import pandas as pd
        df_probs = pd.DataFrame({
            "Grade": [f"Grade {i}" for i in range(5)],
            "진단 등급": [CLASS_INFO[i]["label"] for i in range(5)],
            "예측 확률 (%)": [f"{p * 100:.2f}" for p in probs],
        })
        st.dataframe(df_probs, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### 💾 결과 이미지 저장")

    suptitle = f"Prediction: {info['label']}  |  Confidence: {probs[pred_class]*100:.1f}%"
    dl_bytes = make_download_image(
        img_original, img_gradcam, img_bbox,
        titles=["Original", "Grad-CAM Heatmap", "Original + Bounding Box"],
        suptitle=suptitle,
    )
    st.download_button(
        label="📥 결과 이미지 다운로드 (PNG)",
        data=dl_bytes,
        file_name=f"dr_result_{pred_class}_{uploaded.name}",
        mime="image/png",
    )


if __name__ == "__main__":
    main()