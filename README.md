Bu proje, doğal dil işleme (NLP) tekniklerini kullanarak duygu analizi modelini göstermektedir. Model, metin verilerini pozitif veya negatif duygulara sınıflandırmak için TensorFlow ve Keras kullanılarak oluşturulmuştur.

Proje Yapısı
Proje aşağıdaki ana bileşenleri içermektedir:

Veri Ön İşleme: Metin verilerinin tokenizasyonu ve pad edilmesi.
Model Oluşturma: Keras kullanarak duygu analizi modelinin inşası ve derlenmesi.
Eğitim ve Değerlendirme: Modelin veri seti üzerinde eğitilmesi ve performansının değerlendirilmesi.
Tahmin: Yeni metin verileri üzerinde tahminler yapılması.
Notebook İçeriği
İçe Aktarma ve Kurulum: Gerekli kütüphanelerin içe aktarılması ve ortamın kurulması.
Veri Yükleme: Eğitim ve test için metin verilerinin yüklenmesi.
Veri Ön İşleme:
Tokenizasyon: Metin verilerinin token dizilerine dönüştürülmesi.
Pad Etme: Giriş verileri için uniform dizi uzunluğunun sağlanması.
Model Oluşturma:
Model mimarisinin tanımlanması.
Uygun kayıp fonksiyonu ve optimizör ile modelin derlenmesi.
Model Eğitimi: Modelin ön işlenmiş veriler üzerinde eğitilmesi.
Model Değerlendirme: Modelin test verileri üzerindeki performansının değerlendirilmesi.
Tahmin Yapma: Eğitimli modelin yeni metin girdilerinin duygusunu tahmin etmesi.
