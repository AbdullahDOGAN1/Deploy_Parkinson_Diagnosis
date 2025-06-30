# Gerekli kütüphaneleri içe aktarıyoruz
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io


# --- Model ve Dönüşüm Fonksiyonları (Değişiklik Yok) ---

# Model mimarisini tanımlama fonksiyonu
@st.cache_resource
def get_model_architecture():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


# Görüntü dönüşüm fonksiyonu
def transform_image(image_bytes):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image_transform(image).unsqueeze(0)


# Modeli yükleme fonksiyonu
@st.cache_resource
def load_model():
    # Model dosyasının adını ve yolunu kontrol edin
    model_path = 'parkinson_resnet18_finetuned_BEST.pth'
    model = get_model_architecture()
    try:
        # Modeli CPU üzerinde çalışacak şekilde yüklüyoruz
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Modeli değerlendirme moduna alıyoruz
        return model
    except FileNotFoundError:
        return None


# --- Profesyonel Streamlit Arayüzü ---

# Sayfa yapılandırması
st.set_page_config(
    page_title="MR Görüntüsü Analiz Sistemi",
    page_icon="⚕️",  # Daha profesyonel bir medikal ikon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.title("MR Görüntüsü Analiz Sistemi")
    st.write("---")
    st.subheader("Proje Özeti")
    st.info(
        "Bu uygulama, 2D beyin MR görüntülerinden Parkinson hastalığını teşhis etmek amacıyla "
        "geliştirilmiş bir derin öğrenme modelinin canlı demosudur."
    )

    st.subheader("Model Detayları")
    # Daha temiz bir görünüm için st.markdown kullanıldı
    st.markdown(
        """
        - **Mimari:** `ResNet18` (İnce Ayarlanmış)
        - **Eğitim Veri Seti:** NTUA Parkinson Dataset
        - **Genelleme Testi:** Farklı kaynaklardan gelen veri setleri ile test edilmiştir.
        - **Test Başarısı:** **~%95** Genel Doğruluk
        """
    )

    st.write("---")
    st.subheader("Geliştirici")
    st.text("Abdullah [Soyadınız]")

# --- Ana Sayfa İçeriği ---
st.title("Derin Öğrenme ile Parkinson Hastalığı Tespiti")
st.write(
    "Geliştirilen modeli test etmek için lütfen bir beyin MR görüntüsü yükleyin. "
    "Sistem, yüklediğiniz görüntüyü analiz ederek bir tahmin sunacaktır."
)
st.write("---")

# Model yükleme ve kontrol
model = load_model()

if model is None:
    st.error("Hata: `parkinson_resnet18_finetuned_BEST.pth` model dosyası bulunamadı!")
    st.warning("Lütfen eğitilmiş model dosyasının, `app.py` dosyası ile aynı klasörde olduğundan emin olun.")
else:
    # Dosya yükleme alanı
    uploaded_file = st.file_uploader(
        "Analiz için bir MR görüntüsü seçin",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is None:
        st.info("Lütfen bir MR görüntüsü yükleyerek analizi başlatın.")
    else:
        col1, col2 = st.columns([2, 3])  # Sütun oranları ayarlandı

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

            # Sonuçları gösterme
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
                    f"Model, yüklenen görüntüyü analiz ederek "
                    f"**%{confidence_score * 100:.2f}** olasılıkla **'{prediction}'** "
                    f"sınıfına ait olduğunu tahmin etmiştir. "
                )
                if prediction == 'Parkinson (PD)':
                    st.write(
                        "Bu, görüntüde Parkinson hastalığı ile ilişkilendirilen desenlerin tespit edildiği anlamına gelmektedir.")
                else:
                    st.write(
                        "Bu, görüntüde Parkinson hastalığı ile ilişkilendirilen belirgin desenlerin tespit edilmediği anlamına gelmektedir.")

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

