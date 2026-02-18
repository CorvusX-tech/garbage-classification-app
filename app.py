import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Garbage Classification Deployment",
    page_icon="♻️",
    layout="wide"
)

# รายชื่อคลาส (ต้องตรงกับตอนเทรน)
CLASS_NAMES = ['Battery', 'Biological', 'Brown-glass', 'Cardboard', 'Clothes',
               'Green-glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash', 'White-glass']

# ตรวจสอบ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. MODEL ARCHITECTURES ---
def get_model_structure(model_name, num_classes=12):
    """สร้างโครงสร้างโมเดลเปล่าๆ เพื่อรอโหลด Weight"""
    if model_name == 'EfficientNet_B0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        target_layer = m.features[-1]

    elif model_name == 'MobileNet_V3':
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        target_layer = m.features[-1]

    elif model_name == 'ResNet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target_layer = m.layer4[-1]

    elif model_name == 'ViT':
        m = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        target_layer = None  # ViT ไม่รองรับ Grad-CAM แบบ CNN ปกติ

    elif model_name == 'DenseNet121':
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        target_layer = m.features[-1]

    return m.to(device), target_layer


# --- 3. GRAD-CAM CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output.detach().clone()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().clone()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.features, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# --- 4. HELPER FUNCTIONS ---
def load_model_weights(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return True
    return False


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)
    return conf.item(), idx.item(), probs.cpu().numpy()


# --- 5. MAIN UI ---
def main():
    st.title("♻️ AI Garbage Classification System")
    st.markdown("ระบบจำแนกขยะด้วย AI: เปรียบเทียบระหว่าง **Original Data** และ **Augmented Data**")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    selected_arch = st.sidebar.selectbox(
        "เลือกสถาปัตยกรรมโมเดล (Model Architecture):",
        ['EfficientNet_B0', 'MobileNet_V3', 'ResNet50', 'ViT', 'DenseNet121']
    )

    show_gradcam = st.sidebar.checkbox("Show Grad-CAM (Heatmap)", value=True)
    if selected_arch == 'ViT' and show_gradcam:
        st.sidebar.warning("ViT ไม่รองรับ Grad-CAM ในโหมดนี้")

    # File Uploader
    uploaded_file = st.file_uploader("📂 อัปโหลดรูปภาพขยะ (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert('RGB')

        # แสดงรูปตรงกลาง
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)

        # Preprocess
        img_tensor = preprocess_image(image)

        # --- LOAD MODELS ---
        # 1. Original Model
        model_orig, layer_orig = get_model_structure(selected_arch)
        path_orig = f"model_{selected_arch}_orig.pth"
        loaded_orig = load_model_weights(model_orig, path_orig)

        # 2. Augmented Model
        model_aug, layer_aug = get_model_structure(selected_arch)
        path_aug = f"model_{selected_arch}_aug.pth"
        loaded_aug = load_model_weights(model_aug, path_aug)

        st.markdown("---")

        # --- DISPLAY RESULTS ---
        c1, c2 = st.columns(2)

        # Left Column: Original
        with c1:
            st.info(f"### 📦 Model 1: Original Data")
            if loaded_orig:
                conf, idx, probs = predict(model_orig, img_tensor)
                st.metric("ผลลัพธ์", f"{CLASS_NAMES[idx]}", f"ความมั่นใจ: {conf * 100:.2f}%")
                st.bar_chart({n: p for n, p in zip(CLASS_NAMES, probs)})

                # Grad-CAM Original
                if show_gradcam and layer_orig:
                    gcam = GradCAM(model_orig, layer_orig)
                    mask = gcam.generate(img_tensor, idx)

                    img_np = np.array(image.resize((224, 224))) / 255.0
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    overlay = heatmap * 0.4 + img_np * 0.6
                    st.image(np.clip(overlay, 0, 1), caption="AI Focus Area (Original)")
            else:
                st.error(f"ไม่พบไฟล์: {path_orig}")

        # Right Column: Augmented
        with c2:
            st.success(f"### ✨ Model 2: Augmented Data")
            if loaded_aug:
                conf, idx, probs = predict(model_aug, img_tensor)
                st.metric("ผลลัพธ์", f"{CLASS_NAMES[idx]}", f"ความมั่นใจ: {conf * 100:.2f}%")
                st.bar_chart({n: p for n, p in zip(CLASS_NAMES, probs)})

                # Grad-CAM Augmented
                if show_gradcam and layer_aug:
                    gcam = GradCAM(model_aug, layer_aug)
                    mask = gcam.generate(img_tensor, idx)

                    img_np = np.array(image.resize((224, 224))) / 255.0
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    overlay = heatmap * 0.4 + img_np * 0.6
                    st.image(np.clip(overlay, 0, 1), caption="AI Focus Area (Augmented)")
            else:
                st.error(f"ไม่พบไฟล์: {path_aug}")

        st.caption(
            "Tip: สังเกตว่า Augmented Model มักจะมีความมั่นใจสูงกว่า หรือโฟกัสที่วัตถุได้ชัดเจนกว่า (Grad-CAM สีแดงเข้มที่ตัวขยะ)")


if __name__ == '__main__':
    main()