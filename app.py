# Gerekli kütüphaneleri içe aktarıyoruz
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os
import gdown  # Yeni ve daha güçlü indirme kütüphanesi


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


# --- Modeli Yükleme Fonksiyonu (GÜNCELLENDİ ve BASİTLEŞTİRİLDİ) ---
@st.cache_resource
def load_model():
    model_path = 'parkinson_resnet18_finetuned_BEST.pth'

    # --- ÖNEMLİ: Google Drive Dosya ID'nizi buraya yapıştırın ---
    file_id = '11jw23F_ANuxWQosIGnSy5pqjozGZF7qA'  # Bu örnek ID'yi kendi ID'nizle değiştirin
    # -----------------------------------------------------------

    # Eğer model dosyası yerelde yoksa, gdown ile indir
    if not os.path.exists(model_path):
        with st.spinner(f"Model dosyası indiriliyor... Bu işlem ilk çalıştırmada biraz zaman alabilir."):
            # gdown, Google Drive linkini doğrudan kullanabilir
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            st.success("Model başarıyla indirildi!")

    model = get_model_architecture()
    try:
        # weights_only=False parametresi hala önemli
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None


# --- Streamlit Arayüzü (Değişiklik Yok) ---
st.set_page_config(
    page_title="MR Görüntüsü Analiz Sistemi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("MR Görüntüsü Analiz Sistemi")
    st.write("---")
    st.subheader("Proje Özeti")
    st.info(
        "Bu uygulama, 2D beyin MR görüntülerinden Parkinson hastalığını teşhis etmek amacıyla "
        "geliştirilmiş bir derin öğrenme modelinin canlı demosudur."
    )
    st.subheader("Model Detayları")
    st.markdown(
        """
        - **Mimari:** `ResNet18` (İnce Ayarlanmış)
        - **Eğitim Veri Seti:** NTUA Parkinson Dataset
        - **Test Başarısı:** **~%95** Genel Doğruluk
        """
    )
    st.write("---")
    st.subheader("Geliştirici")
    st.text("Abdullah [Soyadınız]")

st.title("Derin Öğrenme ile Parkinson Hastalığı Tespiti")
st.write(
    "Geliştirilen modeli test etmek için lütfen bir beyin MR görüntüsü yükleyin. "
    "Sistem, yüklediğiniz görüntüyü analiz ederek bir tahmin sunacaktır."
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
    if uploaded_file:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("Yüklenen Görüntü")
            st.image(uploaded_file, caption='Analiz edilecek MR görüntüsü', use_column_width=True)
        with col2:
            st.subheader("Analiz Sonucu")
            with st.spinner('Model görüntüyü analiz ediyor...'):
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
                st.error(f"**Tespit Edilen Durum:** `{prediction}`")
            else:
                st.success(f"**Tespit Edilen Durum:** `{prediction}`")

            st.metric(label="Modelin Güven Skoru", value=f"{confidence_score * 100:.2f}%")
            st.progress(confidence_score)
            with st.expander("Sonuç Detayları"):
                st.write(
                    f"Model, **%{confidence_score * 100:.2f}** olasılıkla görüntünün **'{prediction}'** sınıfına ait olduğunu tahmin etmiştir.")

# Yasal Uyarı
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
