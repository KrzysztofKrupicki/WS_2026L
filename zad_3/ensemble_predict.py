import os
import glob
import math
import gc
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --- IMPORT LOGIKI Z ocr_lib.py ---
from ocr_lib import (
    ENSEMBLE_PRESETS,
    PHASE_CACHE,
    phase_num_beams,
    build_vocabulary,
    correct_text_polish,
    predict_item,
)

# --- KONFIGURACJA ---
VAL_GT_PATH = "easyocr_data/val_gt.txt"
TRAIN_GT_PATH = "easyocr_data/train_gt.txt"
DATA_DIR = "easyocr_data"
NEW_DATA_FOLDER = "new_data"
# Domyslny fallback; per model uzywamy PHASE_CACHE[phase]["batch_size"] (Base 32, Large 16)
BATCH_SIZE = 16

MODEL_PATHS_ALL = {
    "phase5": "./trocr_output_phase5/final_model",
    "phase6": "./trocr_output_phase6/final_model",
}

# preset "1" = phase6 solo, "2" = phase5+6 (~84.2% val)
ENSEMBLE_MODE = "2"

_preset = ENSEMBLE_PRESETS[ENSEMBLE_MODE]
ACTIVE_MODELS = list(_preset["models"])
_p = _preset["params"]
MODEL_WEIGHTS = dict(_p["model_weights"])
TTA_MODE = _p["tta_mode"]
TTA_BIAS = _p["tta_bias"]
TTA_THRESHOLD = _p["tta_threshold"]
# USE_DICTIONARY = False
USE_DICTIONARY = _p.get("use_dictionary", True)

MODEL_PATHS = {k: v for k, v in MODEL_PATHS_ALL.items() if k in ACTIVE_MODELS}


def get_processor():
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                print(f"[+] Wczytywanie procesora lokalnie z: {path}")
                return TrOCRProcessor.from_pretrained(path, local_files_only=True)
            except Exception:
                pass

    print("[+] Wczytywanie procesora z HuggingFace Hub...")
    return TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_cer(predictions, references):
    total_dist = 0
    total_ref_len = 0
    for pred, ref in zip(predictions, references):
        total_dist += levenshtein_distance(pred, ref)
        total_ref_len += len(ref)
    return total_dist / total_ref_len if total_ref_len > 0 else 0.0


def predict_ensemble_single(preds_dict, vocab, word_freqs):
    unique_preds = set()
    for m_data in preds_dict.values():
        if m_data:
            unique_preds.add(m_data["std"][0])
            unique_preds.add(m_data["tta"][0])
    corrected_cache = {
        s: correct_text_polish(s, vocab, word_freqs) for s in unique_preds
    }
    params = dict(_preset["params"])
    params["use_dictionary"] = USE_DICTIONARY
    pred_text, _ = predict_item(
        {"models": preds_dict}, corrected_cache, params, ACTIVE_MODELS
    )
    return pred_text


def run_ensemble_benchmark():
    if not os.path.exists(VAL_GT_PATH):
        print(f"\n[!] Błąd: Nie znaleziono pliku {VAL_GT_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "\n--- TrOCR ADVANCED ENSEMBLE BENCHMARK (OPTIMIZED BATCHING + TTA + DICTIONARY) ---"
    )
    print(f"Urządzenie: {device}")

    print("Ładowanie procesora...", flush=True)
    processor = get_processor()

    # Budujemy słownik ze zbioru treningowego
    vocab, word_freqs = build_vocabulary(TRAIN_GT_PATH)

    available_models = {}
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available_models[key] = path
        else:
            print(f"[!] Ostrzeżenie: Brak modelu {key} pod ścieżką {path}")

    if len(available_models) < 2:
        print(
            "[!] Błąd: Załadowano mniej niż 2 modele. Ensembling wymaga co najmniej 2 modeli!"
        )
        return

    # Wczytanie etykiet
    with open(VAL_GT_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f.readlines() if "\t" in line]

    # Filtrowanie tylko istniejących obrazów
    valid_lines = []
    for img_path, label in lines:
        full_img_path = os.path.join(DATA_DIR, img_path)
        if os.path.exists(full_img_path):
            valid_lines.append((img_path, label))

    total = len(valid_lines)
    # Słowniki na predykcje: predictions[model_key][img_path] = (pred_text, score)
    predictions = {key: {} for key in available_models}
    predictions_tta = {key: {} for key in available_models}

    # Uruchamianie modeli kolejno (oszczędność pamięci VRAM!)
    for model_key, model_path in available_models.items():
        bs = PHASE_CACHE.get(model_key, {}).get("batch_size", BATCH_SIZE)
        num_batches = math.ceil(total / bs)
        print(
            f"\n[+] Ładowanie modelu {model_key} z '{model_path}' do pamięci...",
            flush=True,
        )
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        if device == "cuda":
            model = model.half()
        model.eval()

        beams = phase_num_beams(model_key)
        print(
            f"[+] Generowanie predykcji (Standard + TTA) dla {model_key}, "
            f"batch={bs}, beams={beams}...",
            flush=True,
        )
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"Model {model_key}"):
                batch_items = valid_lines[i * bs : (i + 1) * bs]
                images = []
                images_tta = []
                for img_path, _ in batch_items:
                    full_img_path = os.path.join(DATA_DIR, img_path)
                    img = Image.open(full_img_path).convert("RGB")
                    images.append(img)
                    # TTA: Contrast enhancement (1.2)
                    enhancer = ImageEnhance.Contrast(img)
                    images_tta.append(enhancer.enhance(1.2))

                # Standard
                pixel_values = processor(images, return_tensors="pt").pixel_values.to(
                    device
                )
                if device == "cuda":
                    pixel_values = pixel_values.half()
                outputs = model.generate(
                    pixel_values,
                    num_beams=beams,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                pred_texts = processor.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )
                scores = outputs.sequences_scores.tolist()

                # TTA
                pixel_values_tta = processor(
                    images_tta, return_tensors="pt"
                ).pixel_values.to(device)
                if device == "cuda":
                    pixel_values_tta = pixel_values_tta.half()
                outputs_tta = model.generate(
                    pixel_values_tta,
                    num_beams=beams,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                pred_texts_tta = processor.batch_decode(
                    outputs_tta.sequences, skip_special_tokens=True
                )
                scores_tta = outputs_tta.sequences_scores.tolist()

                for idx, (img_path, _) in enumerate(batch_items):
                    predictions[model_key][img_path] = (
                        pred_texts[idx].strip(),
                        scores[idx],
                    )
                    predictions_tta[model_key][img_path] = (
                        pred_texts_tta[idx].strip(),
                        scores_tta[idx],
                    )

        # Zwolnienie pamięci po zakończeniu predykcji danego modelu
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Analiza wyników ensemblingu
    print("\n[+] Przetwarzanie i porównywanie wyników...", flush=True)
    correct_p5 = 0
    correct_p6 = 0
    correct_opt = 0

    for img_path, label in valid_lines:
        # Zbierz predykcje z każdego modelu dla tego obrazu
        for key in available_models:
            if img_path in predictions[key]:
                pred_text, score = predictions[key][img_path]

                # Liczenie poprawności dla poszczególnych modeli
                if pred_text.lower() == label.lower():
                    if key == "phase5":
                        correct_p5 += 1
                    elif key == "phase6":
                        correct_p6 += 1

        # Zoptymalizowany Ensemble na wagach i TTA wyznaczonych w RAM Search
        preds_structured = {}
        for key in available_models:
            if img_path in predictions[key] and img_path in predictions_tta[key]:
                preds_structured[key] = {
                    "std": predictions[key][img_path],
                    "tta": predictions_tta[key][img_path],
                }
        pred_opt = predict_ensemble_single(preds_structured, vocab, word_freqs)
        if pred_opt.lower() == label.lower():
            correct_opt += 1

    print("\n" + "=" * 80)
    print(f"PORÓWNANIE METOD (Walidacja na {total} próbkach):")
    print("-" * 80)

    if "phase5" in available_models:
        print(
            f"| Model Faza 5 (ALLROUND):               {(correct_p5 / total) * 100:.2f}% ({correct_p5}/{total})"
        )
    if "phase6" in available_models:
        print(
            f"| Model Faza 6 (TrOCR Large):            {(correct_p6 / total) * 100:.2f}% ({correct_p6}/{total})"
        )
    print("-" * 80)
    print(
        f"| ZOPTYMALIZOWANY ENSEMBLE (WAGI/TTA):   {(correct_opt / total) * 100:.2f}% ({correct_opt}/{total})"
    )
    print("=" * 80)


def run_ensemble_folder():
    if not os.path.exists(NEW_DATA_FOLDER):
        os.makedirs(NEW_DATA_FOLDER)

    summary_path = os.path.join(NEW_DATA_FOLDER, "summary.csv")
    gt_labels = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    gt_labels[parts[0]] = parts[1].strip()

    if gt_labels:
        image_paths = []
        for filename in gt_labels.keys():
            full_path = os.path.join(NEW_DATA_FOLDER, filename)
            if os.path.exists(full_path):
                image_paths.append(full_path)
            else:
                print(f"[!] Ostrzeżenie: Plik z summary.csv nie istnieje: {full_path}")
    else:
        image_paths = (
            glob.glob(os.path.join(NEW_DATA_FOLDER, "*.jpg"))
            + glob.glob(os.path.join(NEW_DATA_FOLDER, "*.png"))
            + glob.glob(os.path.join(NEW_DATA_FOLDER, "*.jpeg"))
        )

    if not image_paths:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"\n[+] Analizowanie {len(image_paths)} zdjęć z folderu '{NEW_DATA_FOLDER}' za pomocą Ensemble + TTA + Słownik..."
    )

    processor = get_processor()
    vocab, word_freqs = build_vocabulary(TRAIN_GT_PATH)

    available_models = {}
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available_models[key] = path

    if not available_models:
        print("[!] Błąd: Brak jakichkolwiek modeli do analizy folderu!")
        return

    # Słownik na predykcje folderu: predictions_folder[model_key][img_path] = (pred_text, score)
    predictions_folder = {key: {} for key in available_models}
    predictions_folder_tta = {key: {} for key in available_models}

    for model_key, model_path in available_models.items():
        # Zmniejszamy batch dla phase6 (Large) do 8, aby uniknąć przepełnienia VRAM na kartach 6GB (GTX 1060)
        bs = 16 if model_key == "phase6" else 32
        beams = phase_num_beams(model_key)
        num_batches = math.ceil(len(image_paths) / bs)
        print(
            f"[+] Ładowanie modelu {model_key} do ewaluacji folderu "
            f"(batch={bs}, beams={beams})...",
            flush=True,
        )
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        if device == "cuda":
            model = model.half()
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"Inferencja {model_key}"):
                batch_paths = image_paths[i * bs : (i + 1) * bs]
                images = []
                images_tta = []
                for img_path in batch_paths:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)

                    # TTA: Contrast enhancement (1.2)
                    enhancer = ImageEnhance.Contrast(img)
                    images_tta.append(enhancer.enhance(1.2))

                pixel_values = processor(images, return_tensors="pt").pixel_values.to(
                    device
                )
                if device == "cuda":
                    pixel_values = pixel_values.half()
                outputs = model.generate(
                    pixel_values,
                    num_beams=beams,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                pred_texts = processor.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )
                scores = outputs.sequences_scores.tolist()

                pixel_values_tta = processor(
                    images_tta, return_tensors="pt"
                ).pixel_values.to(device)
                if device == "cuda":
                    pixel_values_tta = pixel_values_tta.half()
                outputs_tta = model.generate(
                    pixel_values_tta,
                    num_beams=beams,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                pred_texts_tta = processor.batch_decode(
                    outputs_tta.sequences, skip_special_tokens=True
                )
                scores_tta = outputs_tta.sequences_scores.tolist()

                for idx, img_path in enumerate(batch_paths):
                    predictions_folder[model_key][img_path] = (
                        pred_texts[idx].strip(),
                        scores[idx],
                    )
                    predictions_folder_tta[model_key][img_path] = (
                        pred_texts_tta[idx].strip(),
                        scores_tta[idx],
                    )

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("-" * 120)
    correct_count = 0
    total_evaluated = 0
    predictions_list = []
    references_list = []

    for img_path in image_paths:
        preds_structured = {}
        for key in available_models:
            preds_structured[key] = {
                "std": predictions_folder[key][img_path],
                "tta": predictions_folder_tta[key][img_path],
            }

        pred_final = predict_ensemble_single(preds_structured, vocab, word_freqs)

        filename = os.path.basename(img_path)
        gt_text = gt_labels.get(filename)

        if gt_text is not None:
            total_evaluated += 1
            is_correct = pred_final.lower() == gt_text.lower()
            if is_correct:
                correct_count += 1
            predictions_list.append(pred_final)
            references_list.append(gt_text)
            status_str = "OK" if is_correct else "BŁĄD"
            print(
                f"| {filename:52} | REAL: {gt_text:20} | ROZPOZNANO: {pred_final:20} | {status_str:5} |"
            )
        else:
            print(f"| {filename:52} | ROZPOZNANO: {pred_final:20} |")

    print("-" * 120)
    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        print(f"\n[+] Wyniki walidacji dla folderu '{NEW_DATA_FOLDER}':")
        print(
            f"    - Przeanalizowano plików z etykietami: {total_evaluated}/{len(image_paths)}"
        )
        print(
            f"    - Poprawne dopasowania (Accuracy): {accuracy:.2f}% ({correct_count}/{total_evaluated})"
        )
        cer_val = calculate_cer(predictions_list, references_list)
        print(f"    - Character Error Rate (CER): {cer_val:.6f}")
        print("=" * 120)

        # Generowanie pliku cache pod optymalizację search_hunt
        import pickle

        new_cache_path = "./predictions_cache_new_data.pkl"
        new_cache_data = {}
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            label = gt_labels.get(filename)
            if label is not None:  # Tylko próbki z etykietami z summary.csv
                new_cache_data[img_path] = {"label": label, "models": {}}
                for key in available_models:
                    new_cache_data[img_path]["models"][key] = {
                        "std": predictions_folder[key][img_path],
                        "tta": predictions_folder_tta[key][img_path],
                    }
        with open(new_cache_path, "wb") as f:
            pickle.dump(new_cache_data, f)
        print(f"\n[+] Zapisano cache dla new_data pod ścieżką: {new_cache_path}")

        print("\n[+] Uruchamianie procedury optymalizacji 'search hunt' na new_data...")
        import ocr_lib
        import random

        random_seeds = [random.randint(0, 100000) for _ in range(5)]
        print(f"    - Wylosowane seedy do optymalizacji: {random_seeds}")
        ocr_lib.search_hunt(
            cache_path=new_cache_path,
            n_seeds=5,
            trials=3000,
            seeds=random_seeds,
            target_names=["1", "2", "3"],
            out_dir="./results_new_data",
        )
    else:
        print(
            f"\n[+] Zakończono analizę folderu '{NEW_DATA_FOLDER}'. Przetworzono {len(image_paths)} plików (brak pliku summary.csv z etykietami)."
        )
        print("=" * 120)


if __name__ == "__main__":
    run_ensemble_benchmark()
    # run_ensemble_folder()
