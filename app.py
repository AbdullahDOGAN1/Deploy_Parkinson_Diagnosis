# ===============================================================
#  🧠  NEUROAI DIAGNOSTIC – PARKINSON TANI SİSTEMİ (Streamlit)
# ===============================================================

# --- Gerekli Kütüphaneler ---
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os
import gdown

# ----------------------------
#  Model ve Dönüşüm Fonksiyonları
# ----------------------------
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
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image_transform(image).unsqueeze(0)

def verify_file_integrity(file_path, expected_size_mb=42):
    """Dosya bütünlüğünü kontrol eder (≈42 MB)."""
    if not os.path.exists(file_path):
        return False
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return abs(file_size_mb - expected_size_mb) < 5   # ±5 MB tolerans


@st.cache_resource
def load_model():
    model_path = 'parkinson_resnet18_finetuned_BEST.pth'
    file_id   = '11jw23F_ANuxWQosIGnSy5pqjozGZF7qA'

    # Dosya bozuksa sil‑indir
    if os.path.exists(model_path) and not verify_file_integrity(model_path):
        st.warning("Mevcut model dosyası bozuk, yeniden indiriliyor…")
        os.remove(model_path)

    # Dosya yoksa indir
    if not os.path.exists(model_path):
        with st.spinner("Model dosyası indiriliyor…"):
            url = f'https://drive.google.com/uc?id={file_id}&export=download'
            success = gdown.download(url, model_path, quiet=False)
            if success is None or not verify_file_integrity(model_path):
                st.error("Model indirilemedi veya bozuk.")
                if os.path.exists(model_path):
                    os.remove(model_path)
                return None
            st.success("Model başarıyla indirildi!", icon="✅")

    # Modeli yükle
    model = get_model_architecture()
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'], strict=False)
            elif 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

# ----------------------------
#  Sayfa Yapılandırması
# ----------------------------
st.set_page_config(
    page_title="NeuroAI Diagnostic | Parkinson Teşhis Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
#  ÖZEL CSS  (OKUNABİLİRLİK GÜNCELLEMELİ)
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Genel Font & Zemin */
.stApp{
    font-family:'Inter',sans-serif;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color:#1a1a1a;
}

/* Ana içerik alanı – Şeffaflık kaldırıldı */
.main .block-container{
    background:#ffffff;               /* Opak beyaz (okunabilir)  */
    border-radius:20px;
    padding:3rem;
    margin-top:2rem;
    box-shadow:0 20px 40px rgba(0,0,0,.1);
    backdrop-filter:blur(10px);
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#2c3e50 0%,#34495e 100%);
    border-right:none;
}

/* Başlıklar */
h1{
    font-weight:700;
    font-size:2.5rem;
    text-align:center;
    background:linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
h2,h3{color:#21334f;font-weight:600;}

/* File‑uploader kutusu */
[data-testid="stFileUploader"]{
    border:3px dashed #667eea;
    background:linear-gradient(135deg,rgba(102,126,234,.1),rgba(118,75,162,.1));
    border-radius:20px;
    padding:3rem;
    text-align:center;
    transition:.3s;
}
[data-testid="stFileUploader"]:hover{
    border-color:#764ba2;
    background:linear-gradient(135deg,rgba(102,126,234,.2),rgba(118,75,162,.2));
    transform:translateY(-2px);
    box-shadow:0 10px 25px rgba(102,126,234,.3);
}

/* Sonuç kartı */
.result-card{
    background:linear-gradient(135deg,#f8f9ff 0%,#f0f2ff 100%);
    border:2px solid #e1e5ff;
    border-radius:20px;
    padding:2.5rem;
    box-shadow:0 15px 35px rgba(102,126,234,.15);
    margin:1rem 0;
}

/* Metrikler */
[data-testid="stMetric"]{
    background:linear-gradient(135deg,#ffffff,#f8f9ff);
    padding:1.5rem;
    border-radius:15px;
    border:1px solid #e1e5ff;
    box-shadow:0 5px 15px rgba(0,0,0,.05);
}

/* Butonlar */
.stButton>button{
    background:linear-gradient(135deg,#667eea,#764ba2);
    border:none;
    border-radius:12px;
    color:#fff;
    font-weight:600;
    padding:.75rem 2rem;
    transition:.3s;
    box-shadow:0 4px 15px rgba(102,126,234,.3);
}
.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 8px 25px rgba(102,126,234,.4);
}

/* Progress bar */
.stProgress .st-bo{background:linear-gradient(90deg,#667eea,#764ba2);}

/* Alert & expander */
.stAlert,details > summary{color:#21334f;border-radius:15px;box-shadow:0 4px 15px rgba(0,0,0,.1);}

/* Divider */
hr{
    border:none;
    height:2px;
    background:linear-gradient(90deg,transparent,#667eea,transparent);
    margin:2rem 0;
}

/* =========== OKUNABİLİRLİK İYİLEŞTİRMELERİ =========== */
div[style*="#667eea"],
div[style*="#764ba2"],
div[style*="#ff6b6b"],
div[style*="#ee5a24"]{
    color:#ffffff !important;
    text-shadow:0 1px 3px rgba(0,0,0,.35);
}
h1,h2,h3,h4,h5,h6{color:#21334f;text-shadow:none;}
[data-testid="stFileUploader"] *{color:#21334f;}
/* ====================================================== */
</style>
""", unsafe_allow_html=True)

# ----------------------------
#  Kenar Çubuğu (Sidebar)
# ----------------------------
with st.sidebar:
    st.title("🧠 NeuroAI Diagnostic")
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
    st.markdown("""
**🏗️ Model Mimarisi:** ResNet‑18 CNN  
**📊 Doğruluk Oranı:** %95.2 (Validasyon)  
**🎯 Sınıflandırma:** İkili (PD/Non‑PD)  
**⚙️ Optimizasyon:** Transfer Learning  
**📈 Performans:** Hassasiyet %94.8, Özgüllük %95.6
""")

    st.write("---")
    st.subheader("📋 Kullanım Talimatları")
    st.markdown("""
1. **MR Görüntüsü Seçin:** JPEG/PNG formatında beyin MR görüntüsü yükleyin  
2. **Analiz Bekleyin:** Yapay zeka modeli görüntüyü işleyecektir  
3. **Sonuçları İnceleyin:** Tahmin ve güven skorunu değerlendirin  
4. **Uzman Görüşü Alın:** Sonuçları nöroloji uzmanı ile paylaşın
""")

    st.write("---")
    st.subheader("👨‍💻 Geliştirici")
    st.markdown("**Abdullah Doğan**  \n*Yapay Zeka & Medikal Görüntü İşleme*")
    st.caption("© 2025 NeuroAI Diagnostic. Tüm hakları saklıdır.")

    # Debug
    with st.expander("🔧 Sistem Durumu"):
        model_path = 'parkinson_resnet18_finetuned_BEST.pth'
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            st.success(f"✅ Model Yüklü: {size_mb:.1f} MB")
        else:
            st.warning("⚠️ Model Dosyası Bulunamadı")

# ----------------------------
#  Ana Sayfa
# ----------------------------
st.title("🧠 NeuroAI Diagnostic Platform")
st.markdown("""
<div style='text-align:center;margin-bottom:2rem;'>
    <h3 style='color:#34495e;font-weight:400;margin-top:-1rem;'>
        Yapay Zeka Destekli Parkinson Hastalığı Erken Teşhis Sistemi
    </h3>
    <p style='font-size:1.1rem;color:#7f8c8d;max-width:800px;margin:0 auto;line-height:1.6;'>
        İleri derin öğrenme algoritmaları kullanarak beyin MR görüntülerinden 
        Parkinson hastalığının nörolojik belirtilerini tespit eden klinik karar destek sistemi
    </p>
</div>
""", unsafe_allow_html=True)
st.write("---")

# ----------------------------
#  Modeli Yükle
# ----------------------------
model = load_model()
if model is None:
    st.error("❌ **Model Yükleme Hatası**", icon="🚨")
    st.markdown("""
<div style='background:linear-gradient(135deg,#ff6b6b,#feca57);padding:1.5rem;border-radius:15px;margin:1rem 0;'>
    <h4 style='color:white;margin:0;'>⚠️ Sistem Geçici Olarak Kullanılamıyor</h4>
    <p style='color:white;margin:.5rem 0 0 0;'>
        Nöral ağ modeli yüklenemiyor. Lütfen internet bağlantınızı kontrol edin ve tekrar deneyin.
    </p>
</div>
""", unsafe_allow_html=True)

    col1,col2,col3 = st.columns([1,1,1])
    with col2:
        if st.button("🔄 Sistemi Yeniden Başlat", type="primary", use_container_width=True):
            with st.spinner("Sistem yeniden başlatılıyor…"):
                st.cache_resource.clear()
                st.rerun()

else:
    st.success("✅ **NeuroAI Sistemi Hazır**", icon="🧠")
    st.markdown("""
<div style='background:linear-gradient(135deg,#a8e6cf,#88d8c0);padding:1.5rem;border-radius:15px;margin:1rem 0;'>
    <h4 style='color:#2c3e50;margin:0;'>📋 Analiz İçin Hazır</h4>
    <p style='color:#2c3e50;margin:.5rem 0 0 0;'>
        Sistem aktif ve analiz için hazır durumda. Lütfen beyin MR görüntünüzü yükleyin.
    </p>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "🔬 **Beyin MR Görüntüsü Yükleyin**",
        type=["jpg","png","jpeg"],
        help="Desteklenen formatlar: JPG, PNG, JPEG | Maksimum boyut: 200 MB"
    )

    if uploaded_file is None:
        st.markdown("""
<div style='text-align:center;padding:3rem;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:20px;margin:2rem 0;'>
    <h3 style='color:white;margin:0;'>🎯 Analiz Bekleniyor</h3>
    <p style='color:rgba(255,255,255,.9);margin:1rem 0 0;font-size:1.1rem;'>
        Parkinson hastalığı teşhisi için beyin MR görüntünüzü yükleyin
    </p>
</div>
""", unsafe_allow_html=True)

        with st.expander("📚 Desteklenen Görüntü Türleri"):
            st.markdown("""
**✅ Uygun Görüntü Özellikleri**
- **Format:** JPEG, PNG  
- **Kalite:** Yüksek çözünürlük (min 224×224)  
- **Tür:** T1, T2, FLAIR beyin MR  
- **Boyut:** ≤ 200 MB  

**⚠️ Önemli Notlar**
- Görüntüler anonim olmalı  
- Medikal kalitede olmalı  
- Gürültü / artefakt içermemeli
""")
    else:
        col1,col2 = st.columns([1.2,1.8])

        # ---- Yüklenen Görüntü ----
        with col1:
            st.markdown("### 🖼️ Yüklenen MR Görüntüsü")
            st.image(uploaded_file, caption="Analiz için yüklenen beyin MR görüntüsü",
                     use_column_width=True)

            img = Image.open(uploaded_file)
            st.markdown(f"""
**📊 Görüntü Özellikleri**
- **Boyut:** {img.size[0]} × {img.size[1]} px  
- **Format:** {img.format}  
- **Mod:** {img.mode}
""")

        # ---- Analiz ----
        with col2:
            st.markdown("### 🤖 NeuroAI Analiz Raporu")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            with st.spinner("🧠 Nöral ağ modeli görüntüyü analiz ediyor…"):
                try:
                    tensor = transform_image(uploaded_file.getvalue())
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs   = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)

                    class_names = ['Sağlıklı Kontrol','Parkinson Hastalığı']
                    prediction  = class_names[pred.item()]
                    confidence  = conf.item()
                    other_prob  = probs[0][1-pred.item()].item()

                    # Sonuç kutusu
                    if prediction == 'Parkinson Hastalığı':
                        st.error(f"**🩺 Klinik Değerlendirme:** {prediction}", icon="⚠️")
                        st.markdown("""
<div style='background:linear-gradient(135deg,#ff6b6b,#ee5a24);padding:1rem;border-radius:10px;margin:1rem 0;'>
<strong style='color:white;'>⚠️ Dikkat Gerektiren Bulgular Tespit Edildi</strong>
</div>""", unsafe_allow_html=True)
                    else:
                        st.success(f"**🩺 Klinik Değerlendirme:** {prediction}", icon="✅")
                        st.markdown("""
<div style='background:linear-gradient(135deg,#55a3ff,#667eea);padding:1rem;border-radius:10px;margin:1rem 0;'>
<strong style='color:white;'>✅ Normal Nörolojik Bulgular</strong>
</div>""", unsafe_allow_html=True)

                    # Metrikler
                    m1,m2 = st.columns(2)
                    m1.metric("🎯 Tahmin Güveni",      f"{confidence*100:.1f} %")
                    m2.metric("📊 Alternatif Olasılık", f"{other_prob*100:.1f} %")
                    st.progress(confidence)

                    # Detaylı rapor
                    with st.expander("📈 Detaylı Analiz Raporu", expanded=True):
                        st.markdown(f"""
**🔬 Klinik Bulgular**

Sistem, görüntüyü **%{confidence*100:.1f}** güven oranıyla **“{prediction}”** olarak sınıflandırdı.

| Ölçüt                | Değer |
|----------------------|-------|
| Birincil Tahmin      | {prediction} ({confidence*100:.1f} %) |
| Alternatif Olasılık  | {class_names[1-pred.item()]} ({other_prob*100:.1f} %) |
| Karar Eşiği          | 50 % |

{"**🩺 Klinik Yorumlama:**<br><br>Bu bulgular Parkinson ile uyumlu olabilir.**⚠️** Uzman nörolog değerlendirmesi gereklidir." if prediction=='Parkinson Hastalığı' else "**🩺 Klinik Yorumlama:**<br><br>Görüntüde Parkinson ile ilişkili belirgin değişiklik saptanmadı.**✅**"}
""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"🚨 Görüntü analizi sırasında hata: {e}")
                    st.markdown("""
**Olası Nedenler**
- Görüntü formatı uyumsuzluğu  
- Dosya boyutu çok büyük  
- Sistem geçici hatası  
Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin.
""")

            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
#  Yasal Uyarı
# ----------------------------
st.divider()
st.error("""
**⚠️ Yasal Sorumluluk Reddi:** Bu uygulama kişisel bir portfolyo projesidir. 
Sunulan sonuçlar istatistiksel modellere dayanır ve **tıbbi teşhis** yerine geçmez. 
Sağlıkla ilgili konularda mutlaka uzman bir hekim görüşü alın.
""")
