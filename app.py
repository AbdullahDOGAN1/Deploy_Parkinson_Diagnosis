# Gerekli kütüphaneleri içe aktarıyoruz
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os
import gdown


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


@st.cache_resource
def load_model():
    model_path = 'parkinson_resnet18_finetuned_BEST.pth'
    file_id = '11jw23F_ANuxWQosIGnSy5pqjozGZF7qA'  # Sizin dosya ID'niz

    if not os.path.exists(model_path):
        with st.spinner(f"Model dosyası indiriliyor... Bu işlem ilk çalıştırmada biraz zaman alabilir."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            st.success("Model başarıyla indirildi!", icon="✅")

    model = get_model_architecture()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None


# --- Profesyonel ve Zenginleştirilmiş Arayüz ---

st.set_page_config(
    page_title="AI Görüntü Analizi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Özel CSS Kodları ---
st.markdown("""
<style>
    /* Genel Font ve Renkler */
    body {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Kenar Çubuğu */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .st-emotion-cache-16txtl3 {
        color: #FAFAFA;
    }
    /* Dosya Yükleme Alanı */
    [data-testid="stFileUploader"] {
        border: 2px dashed #30363D;
        background-color: rgba(48, 54, 61, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
        padding: 2rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #58A6FF;
        background-color: rgba(88, 166, 255, 0.1);
    }
    /* Analiz Sonucu Kartı */
    .result-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    /* Butonlar ve metrikler */
    .stButton>button {
        border-radius: 8px;
    }
    [data-testid="stMetric"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.title("Yapay Zekâ Analiz Sistemi")
    st.write("---")
    st.subheader("Proje Hakkında")
    st.info(
        "Bu web uygulaması, 2D beyin MR görüntülerinden Parkinson hastalığına dair "
        "potansiyel biyobelirteçleri analiz eden bir derin öğrenme modelini sunar.",
        icon="ℹ️"
    )
    st.subheader("Model Detayları")
    st.markdown(
        """
        - **Mimari:** `ResNet18` (İnce Ayarlanmış)
        - **Test Başarısı:** **~%95** Genel Doğruluk
        """
    )
    st.write("---")
    st.subheader("Geliştirici")
    st.text("Abdullah Doğan")
    st.caption("© 2025 - Tüm Hakları Saklıdır.")

# --- Ana Sayfa İçeriği ---
st.title("Derin Öğrenme ile Görüntü Tabanlı Teşhis Analizi")
st.write(
    "Geliştirilen modeli test etmek için lütfen bir beyin MR görüntüsü yükleyin. "
    "Sistem, yüklediğiniz görüntüyü analiz ederek bir değerlendirme sunacaktır."
)
st.write("---")

model = load_model()

if model is None:
    st.error("Hata: Model yüklenemedi!")
else:
    uploaded_file = st.file_uploader(
        "Analiz için bir MR görüntüsü seçin",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is None:
        st.info("Lütfen bir görüntü yükleyerek analizi başlatın.", icon="☝️")
    else:
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Yüklenen Görüntü")
            st.image(uploaded_file, caption='Analiz edilecek MR görüntüsü', use_column_width=True)

        with col2:
            st.subheader("AI Analiz Raporu")

            # Analiz kartını oluşturmak için bir container kullanıyoruz
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                with st.spinner('🤖 Model görüntüyü analiz ediyor...'):
                    image_bytes = uploaded_file.getvalue()
                    tensor = transform_image(image_bytes)

                    with torch.no_grad():
                        outputs = model(tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted_class = torch.max(probabilities, 1)

                class_names = ['Sağlıklı (Non-PD)', 'Parkinson (PD)']
                prediction = class_names[predicted_class.item()]
                confidence_score = confidence.item()

                if prediction == 'Parkinson (PD)':
                    st.error(f"**Model Değerlendirmesi:** `{prediction}`", icon="🩺")
                else:
                    st.success(f"**Model Değerlendirmesi:** `{prediction}`", icon="🩺")

                st.metric(label="Modelin Güven Skoru", value=f"{confidence_score * 100:.2f}%")
                st.progress(confidence_score)

                with st.expander("Sonuç Detayları"):
                    st.write(
                        f"Model, yüklenen görüntüyü analiz ederek "
                        f"**%{confidence_score * 100:.2f}** olasılıkla görüntünün **'{prediction}'** "
                        f"sınıfına ait olduğunu tahmin etmiştir. "
                    )
                    if prediction == 'Parkinson (PD)':
                        st.write(
                            "Bu, görüntüde Parkinson hastalığı ile ilişkilendirilebilen karakteristik özelliklerin tespit edildiği anlamına gelmektedir.")
                    else:
                        st.write(
                            "Bu, görüntüde Parkinson hastalığı ile ilişkilendirilebilen belirgin karakteristik özelliklerin tespit edilmediği anlamına gelmektedir.")

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
