# ✅ Aşağıdaki kod, senin bana attığın Streamlit Parkinson Teşhis uygulamasının tamamıdır.
# 🔁 Gerekli güncellemeler yapıldı:
# - Tema dosyası (.streamlit/config.toml) koddan otomatik oluşturuluyor
# - CSS okunabilirlik sorunları giderildi (özellikle başlıklar, uploader, sidebar)

import os
import io
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import gdown

# -----------------------------------------------------------------------------
# 0. Tema dosyası otomatik oluşturulsun
# -----------------------------------------------------------------------------
THEME_TOML = """
[theme]
base="light"
primaryColor="#667EEA"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2FF"
textColor="#21334F"
font="sans serif"
"""


def ensure_theme():
    path = ".streamlit/config.toml"
    if not os.path.exists(path):
        os.makedirs(".streamlit", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(THEME_TOML.strip())


ensure_theme()


# -----------------------------------------------------------------------------
# 1. Model ve Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_architecture():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def transform_image(image_bytes):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return tfm(img).unsqueeze(0)


def verify_file_integrity(path, expected_size_mb=42):
    return os.path.exists(path) and abs(os.path.getsize(path) / (1024 * 1024) - expected_size_mb) < 5


@st.cache_resource
def load_model():
    path = "parkinson_resnet18_finetuned_BEST.pth"
    file_id = "11jw23F_ANuxWQosIGnSy5pqjozGZF7qA"

    if not verify_file_integrity(path):
        if os.path.exists(path):
            os.remove(path)
        with st.spinner("Model indiriliyor..."):
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            result = gdown.download(url, path, quiet=False)
            if result is None or not verify_file_integrity(path):
                st.error("Model indirilemedi veya bozuk.")
                return None
            st.success("Model başarıyla indirildi.")

    model = get_model_architecture()
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None


# -----------------------------------------------------------------------------
# 2. Sayfa Ayarları ve Tema CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroAI Diagnostic | Parkinson Teşhis Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.main .block-container {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 3rem;
    margin-top: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}
h1, h2, h3, h4, h5, h6 {
    color: #21334F !important;
    font-weight: 600;
}
[data-testid="stFileUploader"] * {
    color: #21334F !important;
}
[data-testid="stSidebar"] * {
    color: #ecf0f1 !important;
}
[data-testid="stMetric"] {
    background: #F8F9FF;
    border: 1px solid #E1E5FF;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 NeuroAI Diagnostic")
    st.markdown("### Parkinson Teşhis Sistemi")
    st.write("---")
    st.subheader("🔬 Sistem Hakkında")
    st.info("Yapay zeka ile Parkinson teşhisi için MR görüntüsü analiz edilir.", icon="🧬")
    st.subheader("⚙️ Teknik Bilgiler")
    st.markdown("""
**Model:** ResNet18  
**Validasyon Doğruluğu:** %95.2  
**Sınıflandırma:** Parkinson / Normal
""")
    st.subheader("👨‍💻 Geliştirici")
    st.markdown("**Abdullah Doğan**
    Yapay
    Zeka
    ve
    Medikal
    Görüntü
    İşleme
    ")
    with st.expander("🔧 Sistem Durumu"):
        path = "parkinson_resnet18_finetuned_BEST.pth"
    if os.path.exists(path):
        st.success(f"Model mevcut: {os.path.getsize(path) / (1024 * 1024):.1f} MB")
    else:
        st.warning("Model dosyası bulunamadı")

    # -----------------------------------------------------------------------------
    # 4. Ana İçerik
    # -----------------------------------------------------------------------------
    st.title("🧠 NeuroAI Diagnostic Platform")
    st.write("---")

    model = load_model()

    if model is None:
        st.error("Model yüklenemedi. Bağlantınızı kontrol edin ve tekrar deneyin.")
    if st.button("Yeniden Dene"):
        st.cache_resource.clear()
    st.rerun()
    st.stop()

    uploaded = st.file_uploader("🔬 Beyin MR Görüntüsü Yükleyin", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("JPEG veya PNG formatında MR yükleyin.")
    st.stop()

    # -----------------------------------------------------------------------------
    # 5. Görüntü Gösterimi ve Analiz
    # -----------------------------------------------------------------------------
    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.image(uploaded, caption="Yüklenen Görüntü", use_column_width=True)
    img = Image.open(uploaded)
    st.markdown(f"**Boyut:** {img.size[0]} x {img.size[1]} px")

    with col2:
        st.markdown("### 🤖 Analiz Sonucu")
    with st.spinner("Analiz yapılıyor..."):
        img_tensor = transform_image(uploaded.getvalue())
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)
    labels = ["Sağlıklı", "Parkinson"]
    st.metric("Tahmin", labels[predicted.item()])
    st.metric("Güven Skoru", f"{confidence.item() * 100:.1f}%")
    st.progress(confidence.item())

    # -----------------------------------------------------------------------------
    # 6. Uyarı
    # -----------------------------------------------------------------------------
    st.divider()
    st.error("**Yasal Uyarı:** Bu araç teşhis amaçlı değildir. Lütfen tıbbi danışmanlık alınız.")
