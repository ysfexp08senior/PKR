# PKR‑Regression: Nedir, Ne İşe Yarar?

Bu belge, **PKR‑Regression** paketinin *neyi bulduğunu* ve *nasıl çalıştığını* sıfırdan anlatır. Amacımız—hiç makine‑öğrenmesi bilmeyen birinin bile paketi neden kullanmak isteyebileceğini* kavratmak.

---
## 1. Sorun: Gürültülü Veriden Anlamlı Tahmin

Çoğu gerçek veri setinde **bazı kayıtlar çok benzer koşullarda benzer sonuçlar** üretirken, diğerleri karma karışık davranır. Geleneksel modeller (linear, tree, NN…) tüm veriyi tek denkleme sığdırmaya çalışır; bu da aşağıdakilere yol açar:

* Karmaşık hiper‑parametre ayarı
* Aşırı uyum (overfitting)
* Anlaşılmaz kara‑kutu tahminler

> ***PKR*** bu problemi tersinden ele alır: **Sadece tartışmasız tutarlı alt‑grupları (kernel) kullan, geri kalan gürültüyü medyana bırak!**

---
## 2. PKR Ne Bulur?

| Terim | Açıklama |
|-------|----------|
| **Kernel (Alt‑Uzay)** | Veri kümesindeki, belirli bir özellik kombinasyonuna uyan küçük hücre. Örn: *Podcast_Name = “Sports Weekly” \∩ Episode_Length 0–0.5 dakika* |
| **Değerli Kernel** | O hücredeki gözlemlerin ≥ %90’ı “düşük” **veya** “yüksek” sonuca sahip + en az 10 satır içerir. |
| **rep0 / rep1 / rep_mid** | Düşük, yüksek ve orta bölgenin temsilci hedef değerleri (sırasıyla %33, %66 ve medyan). |

> **Bulgu:** Her değerli kernel, barındırdığı tüm örneklerde neredeyse *aynı* sonuç görülür—yani güçlü nedensel sinyal taşır.

---
## 3. Nasıl Çalışır? (4 Adım)

1. **Uç Dilim Bayraklama** \> Hedef değeri üç bölgeye ayırır; uçlardaki net sinyalin yerini belirler.
2. **Alt‑Uzay Tarama** \> Özellik kombinasyonlarını (1‑li, 2‑li, 3‑lü) paralel tarar; her hücrede kaç satır var ve ne kadarı düşük/yüksek bakar.
3. **Değerli Kernel Filtresi** \> %90+ aynı etiket + en az 10 satır şartını geçen hücreleri saklar (toplamda ilk 120 taneye kadar*).
4. **Tahmin** \> Bir test satırı, hangi kernel(ler) e düşüyorsa o kernel’in temsilci değer(ler)ini alır. Hiçbiri ise medyan döner.

*Parametreler CLI ile değiştirilebilir: pencere genişliği, min_count, min_ratio, max_dim…*

---
## 4. Neden Kullanasınız?

* **Şeffaf** → Tahmin = açıkça listelenmiş “kural + sayı”.
* **Hızlı** → Model eğitimi, hiper‑parametre yok – sadece tarama & filtre.
* **Genelleştirilebilir** → Farklı veri setlerine sadece tek komutla uyarlanır: `pkr-regression --train my_train.csv --test my_test.csv --target my_y`.
* **Aşırı Uyum Kalkanı** → Gürültülü orta bölge tek bir medyan değeriyle “dengelendiği” için overfitting riski çok düşük.

---
## 5. Bir Bakışta Örnek Kernel

```
--- Kernel 7 ---
Podcast_Name            : Innovators
Episode_Length_minutes  : 0.5 – 1.0
kernel_label            : 1     # yüksek grup
kernel_ratio            : 1.0   # %100 tutarlılık
count                   : 2 311 # sağlam örnek sayısı
```
Bu satır şunu söyler: *“Innovators” podcastinde 0.5–1.0 dk bölüm dinleyen **herkesin** hedefi yüksek çıktı.*

---
## 6. Hızlı Başlangıç
```bash
pip install pkr-regression  # PyPI’den kur
pkr-regression \
  --train  train.csv \
  --test   test.csv \
  --target Listening_Time_minutes
```
* `kernels.csv` → seçilen tüm değerli alt‑uzaylar
* `submission.csv` → Kaggle veya iç rapor tahmin dosyanız

---
**Soru/Cevap & Katkı** → GitHub: <https://github.com/YOUR_USER/pkr-regression>

