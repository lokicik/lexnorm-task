# Kurulum ve Çalıştırma

## Gereksinimler

- Python 3.11 veya 3.12
- Mistral API anahtarı (`MISTRAL_API_KEY`) — Mistral OCR için zorunlu
- Replicate API token'ı (`REPLICATE_API_TOKEN`) — Gemini Flash modeli Replicate üzerinden çalıştırılır; metin temizleme (`extract-target`) ve konsolidasyon LLM aşamalarında kullanılır
- Google Gemini API anahtarı (`GEMINI_API_KEY`) — **opsiyonel**, yalnızca token sayımı için kullanılır (`countTokens` endpoint); verilmezse token sayısı 0 olarak loglanır, işlevsellik etkilenmez
- PDF rasterizasyonu için: `pypdfium2` (pip ile yüklenir) veya sistem kurulu `pdftoppm` / `mutool`

---

## 1. Ortamı Kur

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Windows PowerShell:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> Policy hatası alırsanız: `Set-ExecutionPolicy -Scope Process Bypass`

---

## 2. API Anahtarlarını Ayarla

**macOS / Linux:**
```bash
export MISTRAL_API_KEY="sk-..."        # Mistral OCR (zorunlu)
export REPLICATE_API_TOKEN="r8_..."    # Gemini Flash — metin temizleme + konsolidasyon LLM (opsiyonel)
export GEMINI_API_KEY="AI..."          # Opsiyonel: yalnızca token sayımı için
```

**Windows PowerShell:**
```powershell
$env:MISTRAL_API_KEY      = "sk-..."   # Mistral OCR (zorunlu)
$env:REPLICATE_API_TOKEN  = "r8_..."   # Gemini Flash — metin temizleme + konsolidasyon LLM (opsiyonel)
$env:GEMINI_API_KEY       = "AI..."    # Opsiyonel: yalnızca token sayımı için
```

> **Not:** OCR yalnızca Mistral tarafından yapılır. Gemini Flash, **Replicate** üzerinden (`api.replicate.com/v1/models/google/gemini-3-flash`) iki ayrı aşamada kullanılır: (1) `extract-target` sırasında başka şirket içeriğini temizlemek, (2) `consolidate` ile YK ve esas sözleşme verilerini yapılandırmak (`REPLICATE_API_TOKEN` set edildiğinde otomatik etkinleşir). `GEMINI_API_KEY` yalnızca token sayısını loglamak içindir; verilmezse token değerleri 0 görünür.

`REPLICATE_API_TOKEN` verilmezse pipeline deterministik (regex) modda çalışır, LLM adımları atlanır.

---

## 3. Pipeline'ı Çalıştır

**Tam akış:**
```bash
python -m ttsg pipeline
```

Tam akış şu adımları sırayla çalıştırır:
`ocr (Mistral)` → `extract-target` → `clean (Gemini/Replicate)` → `consolidate` → `export-docx`

> LLM temizleme ve konsolidasyon adımları `REPLICATE_API_TOKEN` ortam değişkeni set edildiğinde otomatik etkinleşir.

---

## 4. Adım Adım Çalıştırma

Yalnızca belirli adımları yeniden çalıştırmak için:

| Komut | Açıklama |
|---|---|
| `python -m ttsg ocr` | Mistral OCR |
| `python -m ttsg extract-target` | Hedef şirket metni ayrıştırma + Gemini temizleme (Replicate) |
| `python -m ttsg extract-target --no-clean` | Gemini temizleme adımını atla |
| `python -m ttsg consolidate` | Konsolidasyon (`REPLICATE_API_TOKEN` set ise LLM, aksi halde regex modu) |
| `python -m ttsg export-docx` | DOCX oluştur |

---

## 5. Testleri Çalıştır

```bash
python -m pytest tests/ -q
```

---



## Troubleshooting

**OCR komutu hata veriyor**
- `.venv` aktif mi? (`source .venv/bin/activate` / `.\.venv\Scripts\Activate.ps1`)
- API anahtarı set edildi mi? (`echo $MISTRAL_API_KEY`)
- Outbound ağ erişimi var mı?

**Rasterization başarısız**
```bash
python -c "import pypdfium2, PIL; print('ok')"
```
Hata alırsanız: `pip install pypdfium2 Pillow`

**`extract-target` hata veriyor veya boş çıktı üretiyor**

OCR cache henüz yok. Önce `python -m ttsg ocr` çalıştırın.

**Windows'ta karakter kodlama hatası**
```powershell
$env:PYTHONUTF8 = "1"
```

---

## Dizin Yapısı

```
data/                  ← kaynak PDF'ler (9 adet)
.cache/                ← rasterize sayfalar + OCR cache (otomatik oluşur)
outputs/
  target-only/         ← her PDF için Parla Enerji metni (MD)
  final/               ← tablolar, esas sözleşme, LLM maliyet dökümü
deliverables/
  ttsg_report.docx     ← teslim dosyası
ttsg/                  ← pipeline kaynak kodu
tests/                 ← unit testler
```
