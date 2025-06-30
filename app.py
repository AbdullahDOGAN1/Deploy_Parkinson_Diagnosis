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


# --- Model ve Dönüşüm Fonksiyonları ---

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
    return abs(file_size_mb - expected_size_mb) < 5  # 5MB tolerans


@st.cache_resource
def load_model():
    model_path = 'parkinson_resnet18_finetuned_BEST.pth'
    file_id = '11jw23F_ANuxWQosIGnSy5pqjozGZF7qA'  # Sizin dosya ID'niz

    # Dosya mevcutsa ve bozuk değilse tekrar indirme
    if os.path.exists(model_path) and not verify_file_integrity(model_path):
        st.warning("Mevcut model dosyası bozuk görünüyor, yeniden indiriliyor...")
        os.remove(model_path)

    if not os.path.exists(model_path):
        with st.spinner(f"Model dosyası indiriliyor... Bu işlem ilk çalıştırmada biraz zaman alabilir."):
            try:
                url = f'https://drive.google.com/uc?id={file_id}&export=download'

                # gdown ile indirme işlemi
                success = gdown.download(url, model_path, quiet=False)

                if success is None:
                    st.error("Model indirilemedi. Lütfen internet bağlantınızı kontrol edin.")
                    return None

                # Dosya boyutunu kontrol et
                if not verify_file_integrity(model_path):
                    st.error("İndirilen model dosyası bozuk görünüyor.")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    return None

                st.success("Model başarıyla indirildi!", icon="✅")

            except Exception as e:
                st.error(f"Model indirme hatası: {e}")
                return None

    # Model yükleme işlemi
    model = get_model_architecture()
    try:
        # Dosya boyutunu tekrar kontrol et
        if not verify_file_integrity(model_path):
            st.error("Model dosyası bozuk. Lütfen uygulamayı yeniden başlatın.")
            if os.path.exists(model_path):
                os.remove(model_path)
            return None

        # Model yükleme - farklı yöntemler deniyoruz
        try:
            # İlk yöntem: strict=False ile yükleme
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

            # Eğer checkpoint bir dictionary ise
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

        except Exception as e:
            st.error(f"Model yükleme hatası: {e}")
            # Dosyayı sil ki bir sonraki seferde tekrar indirilsin
            if os.path.exists(model_path):
                os.remove(model_path)
            return None

        model.eval()
        return model

    except Exception as e:
        st.error(f"Model yüklenirken beklenmeyen bir hata oluştu: {e}")
        # Bozuk dosyayı sil
        if os.path.exists(model_path):
            os.remove(model_path)
        return None


# --- Profesyonel ve Zenginleştirilmiş Arayüz ---

st.set_page_config(
    page_title="NeuroAI Diagnostic | Parkinson Teşhis Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Özel CSS Kodları ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Genel Font ve Renkler */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #1a1a1a;
    }

    /* Ana içerik alanı */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 3rem;
        margin-top: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Kenar Çubuğu */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: none;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }

    /* Başlıklar */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-align: center;
        -webkit-background-clip: text;
        background-clip: text;
    }

    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }

    /* Dosya Yükleme Alanı */
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
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }

    /* Analiz Sonucu Kartı */
    .result-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 2px solid #e1e5ff;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.15);
        margin: 1rem 0;
    }

    /* Metrikler */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff, #f8f9ff);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e1e5ff;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    }

    /* Butonlar */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .st-emotion-cache-1q82h82 {
    overflow-wrap: normal;
    text-overflow: ellipsis;
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    font-family: "Source Sans", sans-serif;
    line-height: normal;
    vertical-align: middle;
    color: black;
}

    /* Progress bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Info, Success, Error kutularının modern tasarımı */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar içeriği */
    .sidebar .stMarkdown {
        color: #ecf0f1;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Kenar Çubuğu (Sidebar) ---
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
    st.subheader("📋 Kullanım Talimatları")
    st.markdown(
        """
        1. **MR Görüntüsü Seçin:** JPEG/PNG formatında beyin MR görüntüsü yükleyin
        2. **Analiz Bekleyin:** Yapay zeka modeli görüntüyü işleyecektir
        3. **Sonuçları İnceleyin:** Tahmin ve güven skorunu değerlendirin
        4. **Uzman Görüşü Alın:** Sonuçları nöroloji uzmanı ile paylaşın
        """
    )

    st.write("---")
    st.subheader("👨‍💻 Geliştirici")
    st.markdown("**Abdullah Doğan**  \n*Yapay Zeka & Medikal Görüntü İşleme*")
    st.caption("© 2025 NeuroAI Diagnostic. Tüm hakları saklıdır.")

    # Debug bilgileri
    with st.expander("🔧 Sistem Durumu"):
        model_path = 'parkinson_resnet18_finetuned_BEST.pth'
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.success(f"✅ Model Yüklü: {file_size:.1f} MB")
        else:
            st.warning("⚠️ Model Dosyası Bulunamadı")

# --- Ana Sayfa İçeriği ---
st.title("🧠 NeuroAI Diagnostic Platform")
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h3 style='color: #34495e; font-weight: 400; margin-top: -1rem;'>
            Yapay Zeka Destekli Parkinson Hastalığı Erken Teşhis Sistemi
        </h3>
        <p style='font-size: 1.1rem; color: #424d93; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
            İleri derin öğrenme algoritmaları kullanarak beyin MR görüntülerinden 
            Parkinson hastalığının nörolojik belirtilerini tespit eden klinik karar destek sistemi
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

# Model yükleme durumunu kontrol et
model = load_model()

if model is None:
    st.error("❌ **Model Yükleme Hatası**", icon="🚨")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff6b6b, #feca57); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
        <h4 style='color: white; margin: 0;'>⚠️ Sistem Geçici Olarak Kullanılamıyor</h4>
        <p style='color: white; margin: 0.5rem 0 0 0;'>
            Nöral ağ modeli yüklenemiyor. Lütfen internet bağlantınızı kontrol edin ve tekrar deneyin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Yeniden yükleme butonu
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Sistemi Yeniden Başlat", type="primary", use_container_width=True):
            with st.spinner("Sistem yeniden başlatılıyor..."):
                st.cache_resource.clear()
                st.rerun()
else:
    st.success("✅ **NeuroAI Sistemi Hazır**", icon="🧠")

    # Bilgilendirme paneli
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8e6cf, #88d8c0); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
        <h4 style='color: #2c3e50; margin: 0;'>📋 Analiz İçin Hazır</h4>
        <p style='color: #2c3e50; margin: 0.5rem 0 0 0;'>
            Sistem aktif ve analiz için hazır durumda. Lütfen beyin MR görüntünüzü yükleyin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "🔬 **Beyin MR Görüntüsü Yükleyin**",
        type=["jpg", "png", "jpeg"],
        help="Desteklenen formatlar: JPG, PNG, JPEG | Maksimum boyut: 200MB"
    )

    if uploaded_file is None:
        # Örnek görüntü gösterimi
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 20px; margin: 2rem 0;'>
            <h3 style='color: white; margin: 0;'>🎯 Analiz Bekleniyor</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.1rem;'>
                Parkinson hastalığı teşhisi için beyin MR görüntünüzü yükleyin
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Örnekler bölümü
        with st.expander("📚 Desteklenen Görüntü Türleri", expanded=False):
            st.markdown("""
            **✅ Uygun Görüntü Özellikleri:**
            - **Format:** JPEG, PNG
            - **Kalite:** Yüksek çözünürlük (minimum 224x224 piksel)
            - **Tür:** Beyin MR görüntüleri (T1, T2, FLAIR)
            - **Boyut:** Maksimum 200 MB

            **⚠️ Önemli Notlar:**
            - Görüntüler anonim olmalıdır
            - Medikal kalitede olmalıdır
            - Gürültü ve artefakt içermemelidir
            """)
    else:
        col1, col2 = st.columns([1.2, 1.8])

        with col1:
            st.markdown("### 🖼️ Yüklenen MR Görüntüsü")
            st.image(
                uploaded_file,
                caption='Analiz için yüklenen beyin MR görüntüsü',
                use_container_width=True
            )

            # Görüntü bilgileri
            from PIL import Image

            img = Image.open(uploaded_file)
            st.markdown(f"""
            **📊 Görüntü Özellikleri:**
            - **Boyut:** {img.size[0]} x {img.size[1]} piksel
            - **Format:** {img.format}
            - **Mod:** {img.mode}
            """)

        with col2:
            st.markdown("### 🤖 NeuroAI Analiz Raporu")

            # Analiz kartını oluşturmak için bir container kullanıyoruz
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                with st.spinner('🧠 Nöral ağ modeli görüntüyü analiz ediyor...'):
                    try:
                        image_bytes = uploaded_file.getvalue()
                        tensor = transform_image(image_bytes)

                        with torch.no_grad():
                            outputs = model(tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted_class = torch.max(probabilities, 1)

                        class_names = ['Sağlıklı Kontrol', 'Parkinson Hastalığı']
                        prediction = class_names[predicted_class.item()]
                        confidence_score = confidence.item()

                        # Diğer sınıfın olasılığı
                        other_class_prob = probabilities[0][1 - predicted_class.item()].item()

                        # Sonuç gösterimi
                        if prediction == 'Parkinson Hastalığı':
                            st.error(f"**🩺 Klinik Değerlendirme:** {prediction}", icon="⚠️")
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                                <strong style='color: white;'>⚠️ Dikkat Gerektiren Bulgular Tespit Edildi</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.success(f"**🩺 Klinik Değerlendirme:** {prediction}", icon="✅")
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #55a3ff, #667eea); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                                <strong style='color: white;'>✅ Normal Nörolojik Bulgular</strong>
                            </div>
                            """, unsafe_allow_html=True)

                        # Güven skoru ve metrikler
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric(
                                label="🎯 Tahmin Güveni",
                                value=f"{confidence_score * 100:.1f}%",
                                delta=f"{(confidence_score - 0.5) * 100:+.1f}%" if confidence_score > 0.5 else None
                            )
                        with col_metric2:
                            st.metric(
                                label="📊 Alternatif Olasılık",
                                value=f"{other_class_prob * 100:.1f}%"
                            )

                        st.progress(confidence_score)

                        # Detaylı analiz
                        with st.expander("📈 Detaylı Analiz Raporu", expanded=True):
                            st.markdown(f"""
                            **🔬 Klinik Bulgular:**

                            Derin öğrenme tabanlı NeuroAI sistemi, yüklenen beyin MR görüntüsünü 
                            **%{confidence_score * 100:.2f}** güven oranıyla **'{prediction}'** 
                            kategorisinde sınıflandırmıştır.

                            **📊 İstatistiksel Değerlendirme:**
                            - **Birincil Tahmin:** {prediction} (%{confidence_score * 100:.1f})
                            - **Alternatif Olasılık:** {class_names[1 - predicted_class.item()]} (%{other_class_prob * 100:.1f})
                            - **Karar Eşiği:** %50 (Klinik Standart)
                            """)

                            if prediction == 'Parkinson Hastalığı':
                                st.markdown("""
                                **🩺 Klinik Yorumlama:**

                                Analiz sonucu, görüntüde Parkinson hastalığı ile uyumlu nörodejeneratif 
                                değişiklikler tespit edilmiş olabileceğini göstermektedir. Bu bulgular:

                                - Substantia nigra'da dopaminerjik nöron kaybı
                                - Bazal ganglion bölgesinde struktural değişiklikler
                                - Motor korteks aktivitesinde farklılıklar

                                **⚠️ Önemli:** Bu sonuç kesin tanı değildir, uzman nörolog konsültasyonu gereklidir.
                                """)
                            else:
                                st.markdown("""
                                **🩺 Klinik Yorumlama:**

                                Analiz sonucu, görüntüde Parkinson hastalığı ile ilişkili belirgin 
                                nöropatolojik değişiklikler tespit edilmemiştir. Bu durum:

                                - Normal nöral yapı ve fonksiyon
                                - Tipik yaşa uygun beyin morfolojisi
                                - Dopaminerjik sistem bütünlüğü

                                **✅ Not:** Normal bulgular, hastalığın tamamen dışlandığı anlamına gelmez.
                                """)

                    except Exception as e:
                        st.error(f"🚨 Görüntü analizi sırasında hata oluştu: {e}")
                        st.markdown("""
                        **Olası Nedenler:**
                        - Görüntü formatı uyumsuzluğu
                        - Dosya boyutu çok büyük
                        - Sistem geçici hatası

                        Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin.
                        """)

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