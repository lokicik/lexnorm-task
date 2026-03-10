# TTSG Pipeline — Parla Enerji Yatırımları A.Ş.

9 adet scan tabanlı Türkiye Ticaret Sicil Gazetesi PDF'inden Parla Enerji Yatırımları Anonim Şirketi'ne ait bilgileri OCR + LLM ve deterministik konsolidasyon yöntemiyle çıkaran Python pipeline'ı.

Tarama (scan) formatındaki PDF'ler önce **Mistral OCR**  modeliyle sayfa başına 5 yatay parçaya (tile) bölünerek yüksek çözünürlüklü görüntüler olarak işlendi; her sayfanın ham metni birleştirildi. Ardından **Gemini 3.0 Flash** (Replicate üzerinden) ile LLM temizleme aşamasında aynı sayfada yer alan diğer şirket ilanları metinden ayıklandı ve yalnızca Parla Enerji'ye ait içerik bırakıldı. Son aşamada esas sözleşme maddeleri, YK olayları ve şirket bilgileri LLM ile yapılandırılarak çıkarıldı; sermaye, MERSİS ve sicil numarası gibi alanlar ek güvence olarak regex ile doğrulandı. Tüm alanlar kaynak PDF ve gazete tarih/sayısına bağlandı; LLM hiçbir zaman bilgi üretmedi, yalnızca OCR metnini yapılandırdı.

Kurulum ve çalıştırma talimatları için → [SETUP.md](SETUP.md)

---

## Çıktı Dosyaları

### `deliverables/`

| Dosya | İçerik |
|---|---|
| `ttsg_report.docx` | **Teslim dosyası.** Tüm tablolar, konsolide esas sözleşme, belge bazlı metinler ve benchmark özeti tek Word belgesi olarak |

### `outputs/final/`

| Dosya | İçerik |
|---|---|
| `company_info.json / .csv / .md` | Güncel şirket bilgileri tablosu — her alan için kaynak PDF + tarih izi |
| `board_members.json / .csv / .md` | Güncel yönetim kurulu — yalnızca aktif üyeler, tüzel kişiler temsil eden gerçek kişi bilgisiyle |
| `articles.json / .md` | Konsolide esas sözleşme — her madde en güncel kaynaktan |
| `consolidation.json` | Tüm konsolidasyon sonucu tek JSON (company_info + board_members + articles) |
| `llm_usage.json` | Konsolidasyon LLM çağrılarının token ve maliyet dökümü — yapılandırma (parse) aşaması |
| `llm_clean_usage.json` | Metin temizleme LLM çağrılarının token ve maliyet dökümü — diğer şirket içeriği silme aşaması |

### `outputs/target-only/`

Her dosya, ilgili PDF'in **yalnızca Parla Enerji'ye ait kısmını**, birebir OCR metni olarak içerir.

| Dosya | Belge |
|---|---|
| `01-2022-12-01-kurulus-ilani.md` | Kuruluş ilanı |
| `02-2023-06-21-esas-sozlesme-degisikligi-sermaye-artirimi.md` | 1. sermaye artırımı |
| `03-2023-10-10-yonetim-kurulu-atama-….md` | 1. YK değişikliği |
| `04-2023-12-29-esas-sozlesme-degisikligi-sermaye-artirimi.md` | 2. sermaye artırımı |
| `05-2024-09-13-yonetim-kurulu-atama-….md` | 2. YK değişikliği |
| `06-2024-10-30-yonetim-kurulu-atama-….md` | 3. YK değişikliği |
| `07-2024-11-05-yonetim-kurulu-atama-….md` | 4. YK değişikliği |
| `08-2025-09-11-…-yonetim-kurulu-ic-yonergesi.md` | YK değişikliği + İç Yönerge |
| `09-2025-12-22-denetci-degisikligi.md` | Denetçi değişikliği |

---

## Mimari ve Yöntem

### İşlem Akışı

```
PDF (scan)
  → Rasterization (pypdfium2 / pdftoppm)
  → OCR (Mistral OCR API)
  → Hedef şirket extraction  [deterministik regex]
  → Metin temizleme  [Gemini Flash via Replicate — opsiyonel]
  → Konsolidasyon  [regex ledger + opsiyonel Gemini parse via Replicate]
  → DOCX export
```

### Filtreleme Yaklaşımı

OCR metninde Parla Enerji başlığı tespit edilip bir sonraki farklı şirket başlığına kadar olan kısım ayrıştırılır. Tek provider olduğundan (Mistral) doğrudan kabul edilir.

### Konsolidasyon Mantığı

- **Şirket bilgileri:** LLM (Gemini Flash via Replicate) birincil kaynak olarak her belgeyi ayrıştırır; MERSIS, sicil müdürlüğü ve sicil numarası gazette header'ından regex ile ek olarak çekilir. `REPLICATE_API_TOKEN` verilmezse pipeline tamamen regex modunda çalışır.
- **YK tablosu:** LLM her belgeden atama/görevden alma eventlerini çıkarır; eventler kronolojik ledger'da işlenerek en güncel aktif üyeler raporlanır.
- **Esas sözleşme:** LLM maddeleri yapılandırılmış olarak çıkarır. Kuruluş ilanı esas alınır; sonraki ilanlardan gelen değişiklikler aynı madde numarası korunarak üzerine yazar. Her madde için kaynak tarih izlenir.

### Sıfır Halüsinasyon İlkesi

- OCR çıktısı dışında bağımsız bilgi üretilmez.
- Her alan için kaynak PDF yolu + yayın tarihi + provider kaydedilir.
- LLM yalnızca OCR metnini yapılandırır; metne bilgi eklemez, alıntı dışına çıkmaz.

### Token Maliyet Optimizasyonu

- OCR bir kez yapılır, `.cache/` içinde kalıcı saklanır; aynı PDF için tekrar çağrı yapılmaz.
- LLM tüm belgeler için çağrılır; ancak madde çıkarımı yalnızca kuruluş ve esas sözleşme değişikliği belgelerinde uygulanır.
- Toplam konsolidasyon LLM maliyeti: ~$0.10 (9 belge) — bkz. `outputs/final/llm_usage.json`

---

## İster Karşılama Özeti

| İster | Durum | Not |
|---|---|---|
| Güncel Şirket Bilgileri Tablosu | ✅ | `company_info.*` + DOCX |
| Yönetim Kurulu Üyeleri Tablosu | ✅ | `board_members.*` + DOCX, tıklanabilir PDF linki dahil |
| Konsolide Esas Sözleşme | ✅ | 16 madde |
| Belge Bazlı Metin Çıkarımı | ✅ | `outputs/target-only/` — 9 PDF ayrıştırılmış |
| Word (DOCX) teslim | ✅ | `deliverables/ttsg_report.docx` |
| Kaynak izlenebilirliği | ✅ | Her alan ve madde için tarih + PDF yolu |
| Halüsinasyon kontrolü | ✅ | LLM yalnızca OCR metnini yapılandırır |
| Token maliyet optimizasyonu | ✅ | Cache + seçici LLM kullanımı, maliyet dökümü DOCX'te |
| Benchmark / kıyas raporu | ✅ | LLM maliyet özeti `llm_usage.json` + DOCX'te tablo |
