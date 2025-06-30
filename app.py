# ===============================================================
#  🧠  NEUROAI DIAGNOSTIC – PARKINSON TANI SİSTEMİ (Streamlit)
#  Tam sürüm – okunabilirlik düzeltmeleri + otomatik tema dosyası
# ===============================================================

"""
Bu dosya, gönderdiğin orijinal kodun **tamamını** içerir ve şu eklemeler/güncellemeler yapılmıştır:
1. `.streamlit/config.toml` dosyası **kod açılışında otomatik** oluşturulur (light tema, koyu metin).
2. CSS yeniden düzenlendi → başlıklar, uploader yazısı, “Sistem Durumu” ve koyu kutulardaki metinler net görünüyor.
3. Python mantığı (model indirme/yükleme, tahmin, UI akışı) **hiç değiştirilmedi**; sadece stil/okunabilirlik eklendi.
"""

# ----------------------------
# 0) Tema dosyasını garanti altına al
# ----------------------------
import os, io
import streamlit as st
from PIL import Image
import torch, gdown
import torch.nn as nn
from torchvision import models, transforms

THEME_TOML = """
[theme]
base="light"
primaryColor="#667EEA"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2FF"
textColor="#21334F"
font="sans serif"
"""

def ensure_theme_config():
    """Çalışma dizininde .streamlit/config.toml yoksa oluşturur."""
    cfg = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(cfg):
        os.makedirs(".streamlit", exist_ok=True)
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(THEME_TOML.strip())
        print("[INFO] Custom theme created at .streamlit/config.toml")

ensure_theme_config()

# ----------------------------
# 1) Model ve Dönüşüm
# ----------------------------
@st.cache_resource
def get_model_architecture():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def transform_image(image_bytes):
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0)

def verify_file_integrity(path, expected_mb=42):
    return os.path.exists(path) and abs(os.path.getsize(path)/(1024*1024)-expected_mb)<5

@st.cache_resource
def load_model():
    path, fid = "parkinson_resnet18_finetuned_BEST.pth", "11jw23F_ANuxWQosIGnSy5pqjozGZF7qA"
    if not verify_file_integrity(path):
        if os.path.exists(path):
            os.remove(path)
        with st.spinner("Model indiriliyor…"):
            url = f"https://drive.google.com/uc?id={fid}&export=download"
            res = gdown.download(url, path, quiet=False)
            if res is None or not verify_file_integrity(path):
                st.error("Model indirilemedi veya bozuk.")
                return None
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

# ----------------------------
# 2) Sayfa ayarı
# ----------------------------
st.set_page_config(
    page_title="NeuroAI Diagnostic | Parkinson Teşhis Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# 3) Okunabilir CSS
# ----------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp{font-family:'Inter',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);}
.main .block-container{background:#FFFFFF;border-radius:20px;padding:3rem;margin-top:2rem;box-shadow:0 20px 40px rgba(0,0,0,.1);} 
[data-testid="stSidebar"]{background:linear-gradient(180deg,#2c3e50 0%,#34495e 100%);} 
[data-testid="stSidebar"] *{color:#ecf0f1 !important;}

/* Başlıklar */
h1,h2,h3,h4,h5,h6{color:#21334F !important;font-weight:600;}

/* File‑uploader */
[data-testid="stFileUploader"]{border:3px dashed #667EEA;background:rgba(102,126,234,.08);border-radius:20px;padding:3rem;text-align:center;}
[data-testid="stFileUploader"] *{color:#21334F !important;}

/* Sonuç kartı & metrikler */
.result-card{background:linear-gradient(135deg,#f8f9ff,#f0f2ff);border:2px solid #e1e5ff;border-radius:20px;padding:2.5rem;margin:1rem 0;box-shadow:0 15px 35px rgba(102,126,234,.15);} 
[data-testid="stMetric"]{background:#F8F9FF;border:1px solid #E1E5FF;border-radius:15px;padding:1.5rem;box-shadow:0 4px 12px rgba(0,0,0,.05);} 

/* Uyarı kutularında beyaz metin */
div[style*="#667eea"],div[style*="#764ba2"],div[style*="#ff6b6b"],div[style*="#ee5a24"]{color:#FFFFFF !important;text-shadow:0 1px 3px rgba(0,0,0,.35);} 
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 4) Sidebar
# ----------------------------
with st.sidebar:
    st.title("🧠 NeuroAI Diagnostic")
    st.markdown("### Parkinson Hastalığı Teşhis Sistemi")
    st.write("---")
    st.subheader("🔬 Sistem Hakkında")
    st.info("Beyin MR görüntülerini analiz ederek erken Parkinson teşhisine yardımcı olur.", icon="🧬")
    st.subheader("⚙️ Teknik Özellikler")
    st.markdown("""**Model:** ResNet‑18\n\n**Validasyon doğruluğu:** %95.2\n\n**Sınıflar:** PD / Non‑PD""")
    st.subheader("👨‍💻 Geliştirici")
    st.markdown("**Abdullah Doğan**  \nYapay Zeka & Medikal Görüntü İşleme")
    with st.expander("🔧 Sistem Durumu"):
        path = "parkinson_resnet18_finetuned_BEST.pth"
        if os.path.exists(path):
            st.success(f"Model dosyası mevcut ({os.path.getsize(path)/(1024*1024):.1f} MB)")
        else:
            st.warning("Model dosyası bulunamadı")

# ----------------------------
# 5) Ana içerik
# ----------------------------
st.title("🧠 NeuroAI Diagnostic Platform")
st.write("---")

model = load_model()
if model is None:
    st.error("Model yüklenemedi – bağlantınızı kontrol edin.")
    if st.button("Yeniden Dene"):
        st.cache_resource.clear(); st.rerun()
    st.stop()

st.success("Sistem hazır – MR yükleyin.")
file = st.file_uploader("🔬 MR Görüntüsü Yükleyin", type=["jpg","jpeg","png"], help="JPG/PNG, max 200 MB")
if not file:
    st.info("MR görüntüsü seçilmedi."); st.stop()

c1,c2 = st.columns([1.2,1.8])
with c1:
    st.image(file, caption="Yüklenen MR", use_column_width=True)
    img = Image.open(file); st.markdown(f"**Boyut:** {img.size[0]}×{img.size[1]} px")
with c2:
    st.markdown("### 🤖 Analiz Sonucu")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    with st.spinner("Analiz yapılıyor…"):
        t = transform_image(file.getvalue())
        out = model(t)
        probs = torch.softmax(out, dim=1)
        conf, cls = torch.max(probs, 1)
        labels = ["Sağlıklı Kontrol","Parkinson Hastalığı"]
    st.metric("Tahmin", labels[cls.item()])
    st.metric("Güven", f"{conf.item()*100:.1f}%")
    st.progress(conf.item())
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 6) Yasal Uyarı
# ----------------------------
st.divider()
st.error("**⚠️ Yasal Uyarı:** Bu uygulama teşhis amacı taşımaz. Lütfen uzman hekime danışın.")
