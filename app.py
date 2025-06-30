# Gerekli kütüphaneleri içe aktarıyoruz
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os
import gdown
import hashlib


# --- Model ve Dönüşüm Fonksiyonları (Değişiklik Yok) ---

@st.cache_resource
def get_model_architecture():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


def transform_image(image_bytes):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image_transform(image).unsqueeze(0)


def verify_file_integrity(file_path, expected_size_mb=42):
    """Dosya bütünlüğünü kontrol eder"""
    if not os.path.exists(file_path):
        return False

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return abs(file_size_mb - expected_size_mb) < 5


@st.cache_resource
def load_model():
    model_path = 'parkinson_resnet18_finetuned_BEST.pth'
    file_id = '11jw23F_ANuxWQosIGnSy5pqjozGZF7qA'

    if os.path.exists(model_path) and not verify_file_integrity(model_path):
        st.warning("Mevcut model dosyası bozuk görünüyor, yeniden indiriliyor...")
        os.remove(model_path)

    if not os.path.exists(model_path):
        with st.spinner(f"Model dosyası indiriliyor..."):
            try:
                url = f'https://drive.google.com/uc?id={file_id}&export=download'
                gdown.download(url, model_path, quiet=False)
                if not verify_file_integrity(model_path):
                    st.error("İndirilen model dosyası bozuk görünüyor.")
                    if os.path.exists(model_path): os.remove(model_path)
                    return None
                st.success("Model başarıyla indirildi!", icon="✅")
            except Exception as e:
                st.error(f"Model indirme hatası: {e}")
                return None

    model = get_model_architecture()
    try:
        if not verify_file_integrity(model_path):
            st.error("Model dosyası bozuk. Lütfen uygulamayı yeniden başlatın.")
            if os.path.exists(model_path): os.remove(model_path)
            return None

        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict') or checkpoint
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            st.error(f"Model yükleme hatası: {e}")
            if os.path.exists(model_path): os.remove(model_path)
            return None

        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenirken beklenmeyen bir hata oluştu: {e}")
        if os.path.exists(model_path): os.remove(model_path)
        return None


# --- Profesyonel ve Zenginleştirilmiş Arayüz ---

st.set_page_config(
    page_title="NeuroAI Diagnostic | Parkinson Teşhis Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Özel CSS Kodları (OKUNABİLİRLİK DÜZELTMELERİ İLE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #1a1a1a;
    }

    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem 3rem 3rem 3rem;
        margin-top: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: none;
    }

    /* OKUNABİLİRLİK İÇİN DÜZELTİLEN BAŞLIKLAR */
    h1 {
        color: #FFFFFF; /* Arkaplanla kontrast için BEYAZ renk */
        font-weight: 700;
        font-size: 2.8rem;
        text-align: center;
        text-shadow: 0 2px 5px rgba(0,0,0,0.2);
        padding-top: 2rem;
    }

    .subtitle {
        color: #E0E6F1; /* Arkaplanla kontrast için AÇIK GRİ renk */
        font-weight: 400;
        font-size: 1.2rem;
        text-align: center;
        max-width: 800px;
        margin: 0.5rem auto 2rem auto;
        line-height: 1.6;
    }

    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    /* DÜZELTME SONU */

    [data-testid="stFileUploader"] {
        border: 3px dashed #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }

    .result-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 2px solid #e1e5ff;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.15);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.title("🧠 NeuroAI Diagnostic")
    # ... (Diğer sidebar içeriği aynı)
    st.markdown("### *Parkinson Hastalığı Teşhis Sistemi*")
    st.write("---")
    st.subheader("🔬 Sistem Hakkında")
    st.info(
        "Bu ileri teknoloji platform, beyin MR görüntülerini derin öğrenme algoritmaları "
        "ile analiz ederek Parkinson hastalığının erken teşhisine yardımcı olan "
        "nörolojik biyobelirteçleri tespit eder.",
        icon="🧬"
    )
    st.subheader("⚡ Teknik Özellikler")
    st.markdown(
        """
        **🏗️ Model Mimarisi:** ResNet-18 CNN  
        **📊 Doğruluk Oranı:** %95.2 (Validasyon)  
        **🎯 Sınıflandırma:** İkili (PD/Non-PD)  
        **⚙️ Optimizasyon:** Transfer Learning  
        **📈 Performans:** Hassasiyet %94.8, Özgüllük %95.6
        """
    )
    st.write("---")
    st.subheader("👨‍💻 Geliştirici")
    st.markdown("**Abdullah Doğan** \n*Yapay Zeka & Medikal Görüntü İşleme*")
    st.caption("© 2025 NeuroAI Diagnostic. Tüm hakları saklıdır.")

# --- Ana Sayfa İçeriği ---
st.title("🧠 NeuroAI Diagnostic Platform")
st.markdown(
    """
    <p class='subtitle'>
        Yapay zeka destekli bu sistem, beyin MR görüntülerinden Parkinson hastalığının 
        nörolojik belirtilerini tespit ederek klinik karar süreçlerine yardımcı olur.
    </p>
    """,
    unsafe_allow_html=True
)

# Ana içeriği beyaz bir konteyner içine alıyoruz
with st.container():
    st.write("---")
    model = load_model()

    if model is None:
        st.error("❌ **Model Yükleme Hatası**", icon="🚨")
        if st.button("🔄 Sistemi Yeniden Başlat", type="primary"):
            st.cache_resource.clear();
            st.rerun()
    else:
        st.success("✅ **NeuroAI Sistemi Hazır**", icon="🧠")
        uploaded_file = st.file_uploader(
            "🔬 **Beyin MR Görüntüsü Yükleyin**",
            type=["jpg", "png", "jpeg"],
            help="Desteklenen formatlar: JPG, PNG, JPEG"
        )
        if uploaded_file:
            col1, col2 = st.columns([1.2, 1.8])
            with col1:
                # ... (Resim gösterme kodu aynı)
                st.markdown("### 🖼️ Yüklenen MR Görüntüsü")
                st.image(uploaded_file, use_column_width=True)
            with col2:
                # ... (Analiz raporu kodu aynı)
                st.markdown("### 🤖 NeuroAI Analiz Raporu")
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    with st.spinner('🧠 Nöral ağ modeli görüntüyü analiz ediyor...'):
                        image_bytes = uploaded_file.getvalue()
                        tensor = transform_image(image_bytes)
                        with torch.no_grad():
                            outputs = model(tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted_class = torch.max(probabilities, 1)
                    class_names = ['Sağlıklı Kontrol', 'Parkinson Hastalığı']
                    prediction = class_names[predicted_class.item()]
                    confidence_score = confidence.item()
                    if prediction == 'Parkinson Hastalığı':
                        st.error(f"**🩺 Klinik Değerlendirme:** {prediction}", icon="⚠️")
                    else:
                        st.success(f"**🩺 Klinik Değerlendirme:** {prediction}", icon="✅")
                    st.metric(label="🎯 Tahmin Güveni", value=f"{confidence_score * 100:.1f}%")
                    st.progress(confidence_score)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Yasal Uyarı Bölümü
    st.divider()
    st.error(
        """
        **⚠️ Yasal Sorumluluk Reddi Beyanı:** Bu uygulama, kişisel bir portfolyo projesi kapsamında geliştirilmiş bir 
        araştırma ve teknoloji demosudur. Sunulan sonuçlar, istatistiksel modellere dayanmaktadır ve 
        **kesinlikle tıbbi bir teşhis niteliği taşımaz.**
        Tıbbi tavsiye, tanı veya tedavi için kullanılamaz. Sağlıkla ilgili herhangi bir endişeniz için 
        lütfen yetkili bir sağlık profesyoneline (doktor) danışınız.
        """
    )
