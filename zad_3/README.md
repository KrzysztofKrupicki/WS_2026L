# zad_3 — TrOCR OCR (PL)

System OCR dedykowany do odczytu pisma odręcznego w języku polskim, oparty na modelach **Microsoft TrOCR** w dwóch wariantach produkcyjnych: **Base ALLROUND** (`phase5`) i **Large ALLROUND** (`phase6`).

---

## 🚀 Wyniki Ostatecznego Benchmarku

Projekt umożliwia bezpośrednie porównanie skuteczności i prędkości obu modeli na zbiorze walidacyjnym (526 linii tekstu) przy użyciu dedykowanego skryptu `benchmark_models.py`.

### Porównanie modeli i ansambli (phase5 vs phase6 vs Ansambl 5+6)

| Model / Ansambl | Architektura | Dokładność (Acc) | CER | Czas całego Val | Przepustowość | Beams |
| --- | --- | --- | --- | --- | --- | --- |
| **phase5** | Base | **79.47%** | 0.10021 | 371.87 s | **1.41 obr/s** | 3 |
| **phase6** | Large | **81.37%** | 0.09084 | 356.26 s | **1.48 obr/s** | 2 |
| **ensemble_5+6** | Ansambl Base+Large | **84.22%** | 0.08407 | 728.13 s | **0.72 obr/s** | 3 (Base) / 2 (Large) |

*Uwaga: Powyższe czasy i przepustowość odzwierciedlają rzeczywisty pomiar na karcie graficznej GTX 1060.*

### Porównanie surowych wyników solo (Standard vs TTA)

Poniższe wyniki (wygenerowane za pomocą komendy `python ocr.py cer`) przedstawiają skuteczność modeli solo bez korekty słownikowej (surowe wyjście z sieci), co pozwala zaobserwować bezpośredni wpływ techniki TTA (kontrastu):

| Model | Dokładność std | CER std | Dokładność TTA | CER TTA | Różnica (TTA vs std) |
| --- | --- | --- | --- | --- | --- |
| **phase5** (Base) | **79.85%** | 0.05986 | **79.09%** | 0.06117 | -0.76 pp |
| **phase6** (Large) | **81.75%** | 0.04555 | **81.94%** | 0.04529 | **+0.19 pp** |

*Wnioski:*
* Dla silniejszego modelu **Large (phase6)** podbicie kontrastu (TTA) przynosi bezpośrednią poprawę skuteczności o **+0.19 pp** celności pełnych słów i dodatkowo obniża CER.
* Dla modelu **Base (phase5)** wariant TTA solo radzi sobie minimalnie gorzej, ale ich sprytne, dynamiczne łączenie na poziomie decyzji ansambla pozwala na wyciągnięcie wyższej celności globalnej.

---

## 🛠️ Komendy CLI (`ocr.py`)

Główny punkt wejścia do aplikacji oferuje 5 komend ułatwiających trening, optymalizację i analizę:

```powershell
python ocr.py cache --phase all          # predykcje phase5 + phase6 → predictions_cache_full.pkl
python ocr.py search                     # Acc presetów 1 i 2 z cache
python ocr.py search --hunt --target 2   # optymalizacja wag presetu Base+Large
python ocr.py search --full --quick      # szybki benchmark kombinacji modeli
python ocr.py cer                        # dokładne metryki CER/Acc na val per model + ansambl
python ocr.py analyze                    # generuje raport błędów oraz pliki CSV w results/
python ocr.py train --size base          # trening ALLROUND Base (trocr_output_phase5)
python ocr.py train --size large         # trening ALLROUND Large (trocr_output_phase6)
```

**Trening na Kaggle (2× T4):**
```bash
torchrun --nproc_per_node=2 ocr.py train --size large
```

---

## 🧩 Presety Ansambla

Wagi są zdefiniowane i zoptymalizowane w `ocr_lib.py` (klucze `"1"` i `"2"` reprezentują tryby ansamblowania):

| Preset | Etykieta | Modele składowe | Dokładność (Acc) | Kiedy stosować |
| --- | --- | --- | --- | --- |
| **1** | solo Large | phase6 | ~**81.9%** | Najszybsza inferencja jednostkowa |
| **2** | Base+Large | phase5 + phase6 | ~**84.2%** | **Produkcja** — najwyższa jakość kosztem czasu |

*Połączenie modeli Base i Large w presecie 2 daje zysk **+2.3 pp** względem solo Large.*

---

## 📖 Architektura i Koncepcja ALLROUND

**ALLROUND** to podejście polegające na treningu na zrównoważonym miksie danych:
* **Pismo realne**: Ręcznie etykietowane skany wycinków (`easyocr_data/train/` oraz `val/`).
* **Dane syntetyczne**: Wygenerowane komputerowo linie tekstu (`synthetic/`) z pełnym pokryciem języka polskiego.
* **Balans**: Wielokrotne powielenie próbek realnych (oversampling), aby dane syntetyczne nie zdominowały procesu uczenia.

### Parametry Inferencji (std vs TTA)
1. **std** – standardowa predykcja na oryginalnym wycinku obrazu.
2. **TTA** (Test-Time Augmentation) – predykcja na obrazie z podbitym kontrastem ×1.2.
3. **Decyzja** – wybierany jest wariant o wyższym log-prob sekwencji (`sequences_score`).
4. **Słownik** – automatyczna korekta literówek na podstawie częstości występowania słów w `train_gt.txt`.

---

## 🏋️ Szczegóły Procesu Uczenia (Treningu)

Trening modeli produkcyjnych odbywa się za pomocą klasy `Seq2SeqTrainer` z biblioteki Hugging Face i jest w pełni konfigurowalny za pomocą CLI.

### 1. Przygotowanie Danych (Data Pipeline)
* **Dataset Mix**: Trening łączy trzy zestawy danych:
  - **Dane realne** (`train_gt.txt`) – ręcznie wycięte i zaetykietowane słowa pisma ręcznego.
  - **Dane syntetyczne** (`synthetic_gt.txt`) – komputerowo generowane próbki z kompletnym pokryciem słownika języka polskiego.
  - **Dane ukierunkowane na diakrytyki** (`train_gt_phase6c.txt`) – specjalny podzbiór do poprawy rozpoznawania polskich znaków (np. ą, ć, ę, ł, ń, ó, ś, ź, ż).
* **Dynamiczny Oversampling**: Aby dane syntetyczne nie zdominowały procesu nauki, próbki realne są powielane (oversampling) za pomocą współczynnika:
  $$\text{factor} = \min\left(8, \max\left(1, \text{round}\left(\frac{\text{len(synth\_df)}}{\text{len(real\_df)}}\right)\right)\right)$$

### 2. Augmentacja Obrazów (Data Augmentation)
W locie stosowany jest agresywny potok augmentacji za pomocą `torchvision.transforms` (klasa `TRAIN_TRANSFORMS` w `ocr_lib.py`), co zwiększa generalizację na nowe charaktery pisma:
* **Rotacja**: Losowy obrót w zakresie $[-7^\circ, 7^\circ]$ (wypełnienie białym tłem).
* **Jitter kolorów**: Zmiana jasności ($\pm 0.2$), kontrastu ($\pm 0.2$) oraz nasycenia ($\pm 0.05$).
* **Elastic Transform**: Elastyczne zniekształcenia (aplikowane z prawdopodobieństwem 25%, $\alpha=12.0, \sigma=5.0$) imitujące rozciąganie papieru.
* **Przekształcenia afiniczne**: Skalowanie ($[0.95, 1.05]$), translacja ($\pm 4\%$) oraz ścinanie (shear $[-4^\circ, 4^\circ]$).
* **Rzut perspektywiczny**: Losowe zniekształcenie perspektywy (aplikowane z prawdopodobieństwem 35%, stopień zniekształcenia $0.12$).
* **Rozmycie Gaussa**: Losowe rozmycie obrazu (aplikowane z prawdopodobieństwem 20%, rozmiar jądra 3, $\sigma \in [0.1, 0.9]$).

### 3. Hiperparametry i Strategia Uczenia
* **Inicjalizacja Wag**:
  - Domyślnie stosowany jest **Warm Start** (`--hf` wyłączone) – trening startuje z wag wyjściowych poprzedniej fazy (`trocr_output_phase5` lub `trocr_output_phase6`), co pozwala na stabilne dotrenowanie na specyficznych danych.
  - Opcjonalnie można trenować od czystych wag Hugging Face (`microsoft/trocr-base-handwritten` lub `large`).
* **Optymalizator i Uczenie**:
  - **Learning Rate (LR)**: $2 \times 10^{-6}$ z harmonogramem typu **Cosine Decay** (łagodne wygaszanie stopy uczenia).
  - **Warmup Ratio**: $0.05$ (początkowe rozgrzewanie LR).
  - **Weight Decay**: $0.08$ (regularyzacja zapobiegająca przeuczeniu).
  - **FP16 (Half Precision)**: Włączone automatycznie na układach GPU wspierających obliczenia zmiennoprzecinkowe 16-bitowe.
* **Walidacja i Stopowanie (Early Stopping)**:
  - Walidacja odbywa się co **200 kroków** na pełnym zbiorze walidacyjnym (`val_gt.txt`).
  - Monitorowaną metryką jest **CER** (Character Error Rate).
  - Stosowany jest `EarlyStoppingCallback` z cierpliwością (patience) wynoszącą **5 epok** dla wersji Base (`phase5`) oraz **6 epok** dla Large (`phase6`). Po zakończeniu przywracany jest najlepszy checkpoint.

### 4. Skalowanie Wydajności i Dystrybucja (Kaggle vs Local)
Potok treningowy automatycznie dostosowuje parametry wykonania w zależności od dostępnego środowiska sprzętowego:
* **Efektywny Rozmiar Batcha**: Niezależnie od środowiska dążymy do efektywnego rozmiaru batcha równego **16** poprzez akumulację gradientów (`gradient_accumulation_steps`):
  - **Lokalnie**:
    - *Base*: batch_size = 4, gradient_accumulation = 4.
    - *Large*: batch_size = 1, gradient_accumulation = 16.
  - **Kaggle (2x T4)**:
    - *Base*: batch_size = 8, gradient_accumulation = 2.
    - *Large*: batch_size = 2, gradient_accumulation = 4 (na GPU).
* **Trening Dystrybuowany (DDP)**: Przy uruchomieniu przez `torchrun` na Kaggle włączana jest obsługa wielu kart graficznych z flagą `ddp_find_unused_parameters` i wyłączeniem interaktywnego paska postępu tqdm.
* **Oszczędność VRAM**: Dla modelu **Large** włączana jest technika **Gradient Checkpointing**, co pozwala na zmieszczenie modelu w ograniczonej pamięci VRAM (np. 15-16 GB na pojedynczym GPU T4).

---

## 📁 Struktura Plików i Katalogów

* `ocr.py` – uproszczona bramka CLI (5 komend).
* `ocr_lib.py` – współdzielona logika, konfiguracja i parametry ansambli.
* `ensemble_predict.py` – skrypt inferencyjny (produkcja: preset `"2"`).
* `benchmark_models.py` – skrypt do natychmiastowego porównania modeli na urządzeniu.
* `easyocr_data/` – zbiór danych OCR (obrazy oraz etykiety `val_gt.txt`).
* `results/` – wygenerowane raporty CER, analizy błędów (`analyze`) i benchmarki.
* `trocr_output_phase5/` & `trocr_output_phase6/` – katalogi z wagami modeli.
