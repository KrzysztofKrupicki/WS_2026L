"""Wspólna logika: cache, benchmark ansamblu, trening ALLROUND."""

import json
import math
import os
import pickle
import random
import itertools
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
    EarlyStoppingCallback,
)
import evaluate

# --- ścieżki ---
CACHE_PATH = "./predictions_cache_full.pkl"
DATA_DIR_DEFAULT = "./easyocr_data"
VAL_GT_REL = "val_gt.txt"
TRAIN_GT_REL = "train_gt.txt"
RESULTS_DIR = "./results"
APPROVED_JSON = "./approved.json"

MODEL_POOL = ["phase5", "phase6"]  # produkcja: presety 1 (6) i 2 (5+6)


def _phase_sort_key(model_name):
    try:
        return int(model_name.replace("phase", ""))
    except ValueError:
        return 99


def sort_models(models):
    """Kolejnosc phase5, phase6 (nie po Acc ani losowo)."""
    return sorted(models, key=_phase_sort_key)


PHASE_CACHE = {
    "phase5": {
        "model_path": "./trocr_output_phase5/final_model",
        "batch_size": 32,
        "num_beams": 3,
        "tokenizer_fallback": "./trocr_output_phase5/final_model",
        "legacy_keys": ("base_allround",),
    },
    "phase6": {
        "model_path": "./trocr_output_phase6/final_model",
        "batch_size": 16,
        "num_beams": 3,
        "tokenizer_fallback": "./trocr_output_phase6/final_model",
        "legacy_keys": (),
    },
}


def cache_batch_size(phase):
    """Batch do cache/inferencji per faza (wynik benchmark_batch_scaling)."""
    return PHASE_CACHE[phase]["batch_size"]


def phase_num_beams(phase):
    """num_beams per faza (domyslnie NUM_BEAMS_DEFAULT)."""
    return PHASE_CACHE[phase].get("num_beams", NUM_BEAMS_DEFAULT)


CACHE_PHASE_CHOICES = ["5", "6", "all"]

# Presety: "1" = solo Large, "2" = Base+Large (ALLROUND)
DEFAULT_ENSEMBLE_PRESET = "2"

# Wagi zsynchronizowane z results/hunt 2026-06-02T16:40:19 (sync_ensemble_presets_from_benchmark)
# Wagi zsynchronizowane z results/hunt 2026-06-02T16:44:56 (sync_ensemble_presets_from_benchmark)
# Wagi zsynchronizowane z results/2026-06-02T16:46:27.576546 (sync_ensemble_presets_from_benchmark)
# Wagi zsynchronizowane z results/hunt 2026-06-03T17:51:13 (sync_ensemble_presets_from_benchmark)
# Wagi zsynchronizowane z results/2026-06-02T16:50:50.104810 (sync_ensemble_presets_from_benchmark)
# Wagi zsynchronizowane z results/2026-06-02T16:50:50.104810 (sync_ensemble_presets_from_benchmark)
ENSEMBLE_PRESETS = {
    "1": {
        "label": 'ansambl 1M (6)',
        "models": ['phase6'],
        "params": {
            "model_weights": {
                "phase6": 2.169096996914096,
            },
            "tta_mode": 'global',
            "tta_bias": -0.0016629949740765149,
            "tta_threshold": -2.238704185026596,
            "use_dictionary": True,
        },
    },
    "2": {
        "label": 'ansambl 2M (5+6)',
        "models": ['phase5', 'phase6'],
        "params": {
            "model_weights": {
                "phase5": -1.538211781204804,
                "phase6": -1.5235063530490582,
            },
            "tta_mode": 'selective',
            "tta_bias": 0.7078217044809696,
            "tta_threshold": -1.2589695276480608,
            "use_dictionary": True,
        },
    },
    "3": {
        "label": 'ansambl 3M (5)',
        "models": ['phase5'],
        "params": {
            "model_weights": {
                "phase5": 1.1221542356535301,
            },
            "tta_mode": 'global',
            "tta_bias": -0.011620826360234204,
            "tta_threshold": -0.9286102914711296,
            "use_dictionary": True,
        },
    },
}

PRESET_SIZES = ("1", "2", "3")
HUNT_ALIASES = {"all": "2", "best": "2"}


def _preset_models_key(models):
    return frozenset(models)


def _benchmark_entry_to_preset(key, entry):
    models = sort_models(entry["models"])
    weights = entry["params"]["model_weights"]
    return {
        "label": _preset_label_for(key, models),
        "models": models,
        "params": {
            "model_weights": {m: weights[m] for m in models},
            "tta_mode": entry["params"]["tta_mode"],
            "tta_bias": entry["params"]["tta_bias"],
            "tta_threshold": entry["params"]["tta_threshold"],
            "use_dictionary": entry["params"].get("use_dictionary", True),
        },
    }


def _format_ensemble_presets_py(presets, benchmark_ts=None):
    ts = benchmark_ts or "benchmark_results.json"
    lines = [
        f"# Wagi zsynchronizowane z results/{ts} (sync_ensemble_presets_from_benchmark)",
        "ENSEMBLE_PRESETS = {",
    ]
    for key in sorted(presets.keys(), key=lambda k: int(k)):
        preset = presets[key]
        lines.append(f'    "{key}": {{')
        lines.append(f'        "label": {preset["label"]!r},')
        lines.append(f'        "models": {sort_models(preset["models"])!r},')
        lines.append('        "params": {')
        p = preset["params"]
        lines.append('            "model_weights": {')
        for mk in sort_models(preset["params"]["model_weights"].keys()):
            mv = preset["params"]["model_weights"][mk]
            lines.append(f'                "{mk}": {mv},')
        lines.append("            },")
        lines.append(f'            "tta_mode": {p["tta_mode"]!r},')
        lines.append(f'            "tta_bias": {p["tta_bias"]},')
        lines.append(f'            "tta_threshold": {p["tta_threshold"]},')
        lines.append(f'            "use_dictionary": {p.get("use_dictionary", True)},')
        lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def _rewrite_ensemble_presets_in_ocr_lib(presets, benchmark_ts=None):
    ocr_lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ocr_lib.py"
    )
    with open(ocr_lib_path, "r", encoding="utf-8") as f:
        text = f.read()
    start = text.find("ENSEMBLE_PRESETS = {")
    if start < 0:
        raise RuntimeError("ENSEMBLE_PRESETS nie znaleziony w ocr_lib.py")
    end = text.find("\n\nPRESET_SIZES", start)
    if end < 0:
        end = text.find("\n\nHUNT_ALIASES", start)
    if end < 0:
        raise RuntimeError("Koniec bloku ENSEMBLE_PRESETS nie znaleziony")
    new_block = _format_ensemble_presets_py(presets, benchmark_ts)
    text = text[:start] + new_block + text[end:]
    with open(ocr_lib_path, "w", encoding="utf-8") as f:
        f.write(text)


def _preset_label_for(key, models):
    nums = "+".join(m.replace("phase", "") for m in models)
    return f"ansambl {key}M ({nums})"


def _eval_preset_acc(cache_data, corrected, preset):
    return run_evaluation(cache_data, corrected, preset["params"], preset["models"])


def _format_acc_delta(old_acc, new_acc):
    if old_acc is None:
        return ""
    d = new_acc - old_acc
    sign = "+" if d >= 0 else ""
    return f" ({sign}{d:.2f} pp)"


def sync_ensemble_presets_from_benchmark(
    json_path=None,
    cache_path=CACHE_PATH,
    apply=True,
    min_gain=0.0,
    verbose=True,
):
    """
    Uaktualnia ENSEMBLE_PRESETS wg results/benchmark_results.json.
    Zwraca liste zmienionych presetow (klucze).
    """
    json_path = json_path or os.path.join(RESULTS_DIR, "benchmark_results.json")
    if not os.path.exists(json_path):
        if verbose:
            print(f"[!] Brak pliku: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    by_size = report.get("by_size") or {}
    ts = report.get("timestamp", "benchmark_results.json")

    cache_data = corrected = None
    if cache_path and os.path.exists(cache_path):
        cache_data = load_synced_cache(cache_path)
        train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
        corrected = build_corrected_cache(cache_data, train_gt)

    if verbose:
        print("\n" + "=" * 70)
        print("SYNC PRESETOW <- benchmark_results.json")
        print("=" * 70)
        if cache_data:
            print(f"  Porownanie: obecne wagi (cache) vs benchmark ({ts[:19]})")
        else:
            print("  [!] Brak cache — tylko acc z benchmarku (bez delty vs obecne)")

    updated = {}
    changes = []
    rows = []

    for key in PRESET_SIZES:
        old = ENSEMBLE_PRESETS.get(key, {})
        preset_models = set(old.get("models", []))
        entry = None
        for size_key, entries in by_size.items():
            for e in entries:
                if set(e["models"]) == preset_models:
                    entry = e
                    break
            if entry:
                break
        if not entry:
            if verbose:
                print(f"  {key:<14} [!] brak wpisu w benchmark")
            continue
        new_acc = entry["acc"]
        old_acc = (
            _eval_preset_acc(cache_data, corrected, old)
            if cache_data
            else old.get("benchmark_acc")
        )
        new_preset = _benchmark_entry_to_preset(key, entry)
        params_differ = (
            old.get("models") != new_preset["models"]
            or old.get("params") != new_preset["params"]
        )
        not_worse = old_acc is None or new_acc >= old_acc - 1e-9
        improved = old_acc is not None and new_acc > old_acc + min_gain
        changed = (params_differ or improved) and not_worse

        if not not_worse:
            status = "pominiety (gorszy)"
        elif changed:
            status = "ZAPISANO"
            updated[key] = new_preset
            changes.append(key)
        else:
            status = "bez zmian"

        rows.append((key, old_acc, new_acc, status))
        if verbose:
            m = " + ".join(new_preset["models"])
            if old_acc is not None:
                print(
                    f"  {key:<14} {old_acc:6.2f}% -> {new_acc:6.2f}%"
                    f"{_format_acc_delta(old_acc, new_acc)}  [{status}]  ({m})"
                )
            else:
                print(f"  {key:<14} benchmark {new_acc:6.2f}%  [{status}]  ({m})")

    if not updated:
        if verbose:
            print("\n  Zadne presety nie wymagaly aktualizacji.")
        return []

    for key, preset in updated.items():
        ENSEMBLE_PRESETS[key] = preset

    if apply and changes:
        _rewrite_ensemble_presets_in_ocr_lib(ENSEMBLE_PRESETS, benchmark_ts=ts)
        if verbose:
            print(f"\n  [+] Zapisano ocr_lib.py: {', '.join(changes)}")
    elif verbose:
        print(f"\n  [i] Do zapisu (apply=False): {', '.join(changes)}")
    return changes


def apply_hunt_to_presets(
    hunt_report,
    cache_path=CACHE_PATH,
    apply=True,
    min_gain=0.01,
    verbose=True,
):
    """Po hunt: porownaj najlepszy wynik per target z presetem w ocr_lib; zapisz jesli lepszy."""
    if not hunt_report or not os.path.exists(cache_path):
        return []

    cache_data = load_synced_cache(cache_path)
    train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
    corrected = build_corrected_cache(cache_data, train_gt)
    targets = hunt_report.get("targets") or {}
    changes = []

    if verbose:
        print("\n[+] Porównanie wyników z obecnymi presetami:")

    for key, data in targets.items():
        if key not in ENSEMBLE_PRESETS:
            continue
        best = data.get("best") or {}
        hunt_acc = best.get("acc")
        if hunt_acc is None:
            continue
        old = ENSEMBLE_PRESETS[key]
        old_acc = _eval_preset_acc(cache_data, corrected, old)
        delta = hunt_acc - old_acc
        models = data.get("models") or old["models"]
        m = " + ".join(models)

        if hunt_acc > old_acc + min_gain:
            status = "ZAPISANO"
            params = best.get("params") or old["params"]
            models = sort_models(models)
            weights = params.get("model_weights") or {}
            ENSEMBLE_PRESETS[key] = {
                "label": _preset_label_for(key, models),
                "models": models,
                "params": {
                    **params,
                    "model_weights": {m: weights[m] for m in models},
                },
            }
            changes.append(key)
        else:
            status = "bez zmian" if abs(delta) <= min_gain else "gorszy"

        if verbose:
            sign = "+" if delta >= 0 else ""
            print(
                f"    - Preset {key} ({m}): {old_acc:.2f}% -> {hunt_acc:.2f}% "
                f"({sign}{delta:.2f} pp) [{status}]"
            )

    if apply and changes:
        _rewrite_ensemble_presets_in_ocr_lib(
            ENSEMBLE_PRESETS,
            benchmark_ts=f"hunt {hunt_report.get('timestamp', '')[:19]}",
        )
        if verbose:
            print(f"\n  [+] Zapisano ocr_lib.py: {', '.join(changes)}")
    elif verbose and not changes:
        print("\n  Zadne presety nie poprawily sie wzgledem obecnych wag.")
    return changes


TRAIN_SIZES = {
    "base": {
        "hf_name": "microsoft/trocr-base-handwritten",
        "warm_path": "./trocr_output_phase5/final_model",
        "archive_warm": "./trocr_output_base_allround_warm/final_model",
        "out_warm": "./trocr_output_phase5",
        "out_hf": "./trocr_output_phase5_hf",
        "weight_hints": ("phase5", "base", "trocr"),
        "gradient_checkpointing": False,
        "early_stopping": 5,
    },
    "large": {
        "hf_name": "microsoft/trocr-large-handwritten",
        "warm_path": "./trocr_output_phase6/final_model",
        "archive_warm": None,
        "out_warm": "./trocr_output_phase6",
        "out_hf": "./trocr_output_phase6_hf",
        "weight_hints": ("phase6", "large", "trocr"),
        "gradient_checkpointing": True,
        "early_stopping": 6,
    },
}

IMAGE_FOLDER_LOOKUP = {}
NUM_BEAMS_DEFAULT = 3
NUM_BEAMS = NUM_BEAMS_DEFAULT  # alias wsteczny
TRIALS_PER_COMB = 2000
TRIALS_QUICK = 150
TRIALS_SOLO_OPT = 500
TRIALS_HUNT = 3000
HUNT_SEEDS = [42, 0, 1, 7, 123, 2025, 31415, 9999, 2718, 65536]


def resolve_hunt_targets(target_names=None):
    """Zwraca liste (key, label, models). target_names=None -> presety 1 i 2."""
    registry = {}
    for key in PRESET_SIZES:
        preset = ENSEMBLE_PRESETS[key]
        registry[key] = (preset["label"], list(preset["models"]))
    for alias, key in HUNT_ALIASES.items():
        if key in registry:
            registry[alias] = registry[key]

    if not target_names:
        return [(k, registry[k][0], registry[k][1]) for k in PRESET_SIZES]

    out = []
    for raw in target_names:
        name = raw.strip().lower()
        if name in HUNT_ALIASES:
            name = HUNT_ALIASES[name]
        if name not in registry:
            known = ", ".join(sorted(registry.keys()))
            print(f"[!] Nieznany --target: {raw!r}. Dostepne: {known}")
            continue
        label, models = registry[name]
        out.append((name, label, models))
    return out


HUNT_TARGET_CHOICES = sorted(set(PRESET_SIZES) | set(HUNT_ALIASES.keys()))

# --- słownik / ewaluacja ansamblu ---


def strip_polish_diacritics(s):
    return s.translate(str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ"))


def build_vocabulary(train_gt_path):
    vocab = set()
    word_frequencies = defaultdict(int)
    if os.path.exists(train_gt_path):
        with open(train_gt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    text = parts[1].strip()
                    vocab.add(text)
                    vocab.add(text.lower())
                    word_frequencies[text.lower()] += 10
                    for w in text.split():
                        w_clean = w.strip(".,;:!?()")
                        if len(w_clean) > 1:
                            vocab.add(w_clean)
                            vocab.add(w_clean.lower())
                            word_frequencies[w_clean.lower()] += 1
    return vocab, word_frequencies


def correct_word(word, vocab, word_frequencies):
    word_lower = word.lower()
    if word_lower in vocab or len(word) <= 3:
        return word
    word_stripped = strip_polish_diacritics(word_lower)
    candidates = [
        v for v in vocab if strip_polish_diacritics(v.lower()) == word_stripped
    ]
    if candidates:
        best = max(candidates, key=lambda c: word_frequencies.get(c.lower(), 0))
        return best.capitalize() if word[0].isupper() else best
    return word


def correct_text_polish(text, vocab, word_frequencies):
    text_lower = text.strip().lower()
    if text_lower in vocab:
        for v_word in vocab:
            if v_word.lower() == text_lower:
                return v_word
    words = text.split()
    corrected = []
    for w in words:
        stripped_w = w.strip(".,;:!?()")
        if stripped_w:
            c = correct_word(stripped_w, vocab, word_frequencies)
            idx = w.find(stripped_w)
            corrected.append(
                w[:idx] + c + w[idx + len(stripped_w) :] if idx != -1 else c
            )
        else:
            corrected.append(w)
    return " ".join(corrected)


def predict_item(item, corrected_cache, params, models_to_use):
    """Jedna predykcja ansambla dla próbki (tekst po TTA/wagach/słowniku)."""
    weights = params["model_weights"]
    tta_bias = params.get("tta_bias", 0.0)
    tta_mode = params["tta_mode"]
    tta_threshold = params["tta_threshold"]
    use_dict = params["use_dictionary"]
    preds_for_vote = {}
    for m_key in models_to_use:
        m_data = item["models"].get(m_key)
        if not m_data or m_key not in weights:
            continue
        std_text, std_score = m_data["std"]
        tta_text, tta_score = m_data["tta"]
        m_weight = weights.get(m_key, 0.0)
        std_w = std_score + m_weight
        tta_w = tta_score + m_weight + tta_bias
        if tta_mode == "global":
            preds_for_vote[m_key] = (
                (tta_text, tta_w) if tta_w > std_w else (std_text, std_w)
            )
        elif tta_mode == "selective":
            if std_score < tta_threshold:
                preds_for_vote[m_key] = (
                    (tta_text, tta_w) if tta_w > std_w else (std_text, std_w)
                )
            else:
                preds_for_vote[m_key] = (std_text, std_w)
        else:
            preds_for_vote[m_key] = (std_text, std_w)
    if not preds_for_vote:
        return "", None
    best_key = max(preds_for_vote, key=lambda k: preds_for_vote[k][1])
    pred_text = preds_for_vote[best_key][0]
    if use_dict:
        pred_text = corrected_cache.get(pred_text, pred_text)
    return pred_text, best_key


def model_std_pred(item, model_key, corrected_cache, use_dict=True):
    m = item["models"].get(model_key)
    if not m:
        return ""
    text = m["std"][0]
    if use_dict:
        text = corrected_cache.get(text, text)
    return text


def run_evaluation(cache_data, corrected_cache, params, models_to_use):
    correct = 0
    total = len(cache_data)
    for item in cache_data.values():
        pred_text, _ = predict_item(item, corrected_cache, params, models_to_use)
        if pred_text.lower() == item["label"].lower():
            correct += 1
    return (correct / total) * 100


def relative_time(models):
    return sum(2.0 if m in ("phase6", "phase6c") else 1.0 for m in models)


def load_synced_cache(cache_path=CACHE_PATH, val_gt_path=None):
    val_gt_path = val_gt_path or os.path.join(DATA_DIR_DEFAULT, VAL_GT_REL)
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    val_labels = {}
    with open(val_gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                val_labels[parts[0]] = parts[1].strip()
    synced = {}
    for img_path, label in val_labels.items():
        if img_path in cache_data:
            cache_data[img_path]["label"] = label
            synced[img_path] = cache_data[img_path]
    if not synced:
        return cache_data
    return synced


def build_corrected_cache(cache_data, train_gt_path):
    vocab, word_freqs = build_vocabulary(train_gt_path)
    unique_preds = set()
    for item in cache_data.values():
        for m_data in item["models"].values():
            if m_data:
                unique_preds.add(m_data["std"][0])
                unique_preds.add(m_data["tta"][0])
    return {s: correct_text_polish(s, vocab, word_freqs) for s in unique_preds}


def analyze_errors(cache_path=CACHE_PATH, out_dir=RESULTS_DIR):
    """
    Analiza błędów na val: per model, competition, podejrzenia błędnych etykiet GT.
    Zapis: results/error_analysis_report.txt, errors_suspect_gt.csv, errors_competition.csv
    """
    import csv

    cache_data = load_synced_cache(cache_path)
    train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
    corrected = build_corrected_cache(cache_data, train_gt)
    models = [
        m
        for m in MODEL_POOL
        if any(m in it.get("models", {}) for it in cache_data.values())
    ]

    comp = ENSEMBLE_PRESETS[DEFAULT_ENSEMBLE_PRESET]
    rows = []
    for img_path, item in cache_data.items():
        label = item["label"]
        comp_pred, comp_winner = predict_item(
            item, corrected, comp["params"], comp["models"]
        )
        per_model = {m: model_std_pred(item, m, corrected) for m in models}
        votes = Counter(per_model.values())
        majority, maj_n = votes.most_common(1)[0] if votes else ("", 0)
        n_correct = sum(1 for p in per_model.values() if p.lower() == label.lower())
        diac_only = comp_pred.lower() != label.lower() and strip_polish_diacritics(
            comp_pred.lower()
        ) == strip_polish_diacritics(label.lower())
        unanimous = (
            len(set(per_model.values())) == 1
            and comp_pred.lower() != label.lower()
            and maj_n >= len(models) - 1
        )
        maj_need = max(2, (len(models) * 2 + 2) // 3)
        suspect_gt = unanimous or (
            maj_n >= maj_need and majority.lower() != label.lower()
        )
        rows.append(
            {
                "img": img_path,
                "label": label,
                "competition": comp_pred,
                "winner": comp_winner or "",
                "majority": majority,
                "maj_count": maj_n,
                "n_models_ok": n_correct,
                "suspect_gt": suspect_gt,
                "unanimous_wrong": unanimous,
                "diacritics_only": diac_only,
                "comp_ok": comp_pred.lower() == label.lower(),
                **{f"p_{m}": per_model.get(m, "") for m in models},
            }
        )

    total = len(rows)
    comp_ok = sum(1 for r in rows if r["comp_ok"])
    suspect = [r for r in rows if r["suspect_gt"]]
    unanimous_list = [r for r in rows if r["unanimous_wrong"]]
    diac_list = [r for r in rows if r["diacritics_only"]]
    all_wrong = [r for r in rows if r["n_models_ok"] == 0]
    comp_err = [r for r in rows if not r["comp_ok"]]

    lines = [
        "=" * 80,
        f"ANALIZA BŁĘDÓW VAL — {total} próbek",
        "=" * 80,
        "",
        "--- Dokładność (STD + słownik) ---",
    ]
    for m in models:
        ok = sum(1 for r in rows if r.get(f"p_{m}", "").lower() == r["label"].lower())
        lines.append(f"  {m:8} {100 * ok / total:6.2f}%  ({ok}/{total})")
    lines += [
        "",
        f"competition (prod): {100 * comp_ok / total:.2f}%  ({comp_ok}/{total})",
        "",
        "--- Kategorie ---",
        f"  Wszystkie modele źle (0/{len(models)}):  {len(all_wrong):4}  ({100 * len(all_wrong) / total:.1f}%)",
        f"  competition źle:                {len(comp_err):4}  ({100 * len(comp_err) / total:.1f}%)",
        f"  Tylko diakrytyki (comp!=GT):     {len(diac_list):4}  ({100 * len(diac_list) / total:.1f}%)",
        f"  Jednomyślne modele != GT:         {len(unanimous_list):4}  ({100 * len(unanimous_list) / total:.1f}%)",
        f"  Podejrzenie błędnego GT:         {len(suspect):4}  ({100 * len(suspect) / total:.1f}%)",
        "",
        "Podejrzenie GT = 4+ modele zgodne na innym tekscie LUB wszystkie modele",
        "na tym samym błędnym słowie. Warto ręcznie obejrzeć obrazy w CSV.",
        "",
        "--- TOP 20: podejrzenie błędnego GT (obraz | GT | większość modeli) ---",
    ]
    for r in sorted(suspect, key=lambda x: -x["maj_count"])[:20]:
        lines.append(
            f"  {r['img']}\n    GT: {r['label']!r}  |  modele ({r['maj_count']}/{len(models)}): {r['majority']!r}"
            f"  |  comp: {r['competition']!r}"
        )
    lines += ["", "--- TOP 15: competition źle, phase6 trafione ---"]
    p6_fix = [
        r for r in comp_err if r.get("p_phase6", "").lower() == r["label"].lower()
    ][:15]
    for r in p6_fix:
        lines.append(
            f"  {r['img']}\n    GT: {r['label']!r}  comp: {r['competition']!r}  phase6: {r['p_phase6']!r}"
        )
    lines += ["", "--- TOP 15: trudne (0 modeli trafia) ---"]
    for r in all_wrong[:15]:
        preds = " | ".join(f"{m}={r.get(f'p_{m}', '')!r}" for m in models[:4])
        lines.append(f"  {r['img']}\n    GT: {r['label']!r}  {preds}")

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "error_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    def write_csv(path, data, fieldnames):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)

    base_fields = [
        "img",
        "label",
        "competition",
        "winner",
        "majority",
        "maj_count",
        "n_models_ok",
        "suspect_gt",
        "unanimous_wrong",
        "diacritics_only",
        "comp_ok",
    ] + [f"p_{m}" for m in models]
    write_csv(os.path.join(out_dir, "errors_suspect_gt.csv"), suspect, base_fields)
    write_csv(os.path.join(out_dir, "errors_competition.csv"), comp_err, base_fields)
    write_csv(os.path.join(out_dir, "errors_all_samples.csv"), rows, base_fields)

    preview = "\n".join(lines[:45]).encode("ascii", errors="replace").decode("ascii")
    print(preview)
    print(f"\n... pelny raport: {report_path}")
    print(
        f"    CSV do przeglądu GT: {out_dir}/errors_suspect_gt.csv ({len(suspect)} wierszy)"
    )
    print(
        f"    Wszystkie błędy comp: {out_dir}/errors_competition.csv ({len(comp_err)} wierszy)"
    )


def _phase_solo_std_acc(cache_data, phase, use_dict=False, train_gt=None):
    """Dokładność solo (std) jednej fazy na zsynchronizowanym cache."""
    if use_dict:
        train_gt = train_gt or os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
        corrected = build_corrected_cache(cache_data, train_gt)
    ok = 0
    n = 0
    for item in cache_data.values():
        m = item["models"].get(phase)
        if not m:
            continue
        n += 1
        pred = m["std"][0]
        if use_dict:
            pred = corrected.get(pred, pred)
        if pred.lower() == item["label"].lower():
            ok += 1
    return (100.0 * ok / n, ok, n) if n else (0.0, 0, 0)


def report_val_cer(cache_path=CACHE_PATH, out_dir=RESULTS_DIR, sync_benchmark=True):
    """CER na val (526) per model z cache — std i TTA, plus ansambl (presety 1 i 2).

    Zapisuje results/cer_report.json. Nie optymalizuje wag.
    Gdy sync_benchmark=True i jest benchmark_results.json — na koncu sync presetow.
    """
    if not os.path.exists(cache_path):
        print(f"[!] Brak {cache_path}")
        return

    cache_data = load_synced_cache(cache_path)
    models = [
        m
        for m in MODEL_POOL
        if any(m in it.get("models", {}) for it in cache_data.values())
    ]
    if not models:
        print("[!] Brak modeli w cache.")
        return

    cer_metric = evaluate.load("cer")
    train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
    corrected = build_corrected_cache(cache_data, train_gt)

    refs = [item["label"] for item in cache_data.values()]
    n = len(refs)

    def cer_acc(preds):
        c = cer_metric.compute(predictions=preds, references=refs)
        exact = sum(1 for p, r in zip(preds, refs) if p.lower() == r.lower())
        return c, 100.0 * exact / n

    rows = []
    print(f"[i] Val: {n} probek | CER (nizszy = lepiej)\n")
    print(
        f"{'Model':<10} | {'CER std':<10} | {'Acc std':<10} | {'CER TTA':<10} | {'Acc TTA':<10}"
    )
    print("-" * 58)

    for m in models:
        ps, pt = [], []
        for item in cache_data.values():
            md = item["models"].get(m)
            if not md:
                continue
            ps.append(md["std"][0])
            pt.append(md["tta"][0])
        if len(ps) != n:
            print(f"[!] {m}: tylko {len(ps)}/{n} probek w cache")
            continue
        cer_s, acc_s = cer_acc(ps)
        cer_t, acc_t = cer_acc(pt)
        rows.append(
            {
                "model": m,
                "cer_std": cer_s,
                "acc_std_pct": acc_s,
                "cer_tta": cer_t,
                "acc_tta_pct": acc_t,
                "n": n,
            }
        )
        print(
            f"{m:<10} | {cer_s:<10.6f} | {acc_s:<9.4f}% | {cer_t:<10.6f} | {acc_t:<9.4f}%"
        )

    # ansambl (predykcja jak w search — ze slownikiem)
    print("-" * 58)
    print("Ansambl (pred + slownik PL):")
    for key in sorted(PRESET_SIZES, key=int):
        preset = ENSEMBLE_PRESETS[key]
        preds = []
        for item in cache_data.values():
            p, _ = predict_item(item, corrected, preset["params"], preset["models"])
            preds.append(p)
        cer_v, acc_v = cer_acc(preds)
        rows.append(
            {
                "model": f"ensemble:{key}",
                "cer_std": cer_v,
                "acc_std_pct": acc_v,
                "cer_tta": None,
                "acc_tta_pct": None,
                "n": n,
            }
        )
        print(f"  {preset['label']:<20} | CER {cer_v:.6f} | Acc {acc_v:.4f}%")

    os.makedirs(out_dir, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "val_samples": n,
        "cache_path": cache_path,
        "per_model": rows,
    }
    txt_path = os.path.join(out_dir, "cer_report.txt")
    json_path = os.path.join(out_dir, "cer_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"CER val ({n} probek) — {report['timestamp']}\n\n")
        f.write(
            f"{'Model':<22} | {'CER std':<10} | {'Acc std':<10} | {'CER TTA':<10} | {'Acc TTA'}\n"
        )
        f.write("-" * 70 + "\n")
        for r in rows:
            if r["model"].startswith("ensemble:"):
                continue
            f.write(
                f"{r['model']:<22} | {r['cer_std']:<10.6f} | {r['acc_std_pct']:.4f}%   | "
                f"{r['cer_tta']:<10.6f} | {r['acc_tta_pct']:.4f}%\n"
            )
        f.write("\nAnsambl:\n")
        for r in rows:
            if not r["model"].startswith("ensemble:"):
                continue
            f.write(
                f"  {r['model']:<20} CER {r['cer_std']:.6f}  Acc {r['acc_std_pct']:.4f}%\n"
            )
    print(f"\n[+] Zapisano: {txt_path}\n[+] {json_path}")

    bench_path = os.path.join(out_dir, "benchmark_results.json")
    if sync_benchmark and os.path.exists(bench_path):
        print("\n[i] Benchmark na dysku — sync presetow z najlepszych kombinacji:")
        sync_ensemble_presets_from_benchmark(
            json_path=bench_path, cache_path=cache_path, apply=True, verbose=True
        )
    elif sync_benchmark:
        print(
            "\n[i] Brak results/benchmark_results.json — presety bez zmian "
            "(najpierw: python ocr.py search --full)"
        )

    return report


def search_presets(cache_path=CACHE_PATH):
    if not os.path.exists(cache_path):
        print(f"[!] Brak {cache_path}")
        return
    cache_data = load_synced_cache(cache_path)
    train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
    corrected = build_corrected_cache(cache_data, train_gt)
    print(f"[i] Val: {len(cache_data)} próbek")
    if os.path.exists(cache_path):
        print(
            f"[i] Cache: {cache_path} ({datetime.fromtimestamp(os.path.getmtime(cache_path))})"
        )
    p6 = PHASE_CACHE["phase6"]["model_path"]
    if os.path.exists(p6):
        mt = max(
            os.path.getmtime(os.path.join(p6, f))
            for f in os.listdir(p6)
            if os.path.isfile(os.path.join(p6, f))
        )
        acc, ok, n = _phase_solo_std_acc(cache_data, "phase6")
        print(
            f"[i] phase6 wagi: {p6} ({datetime.fromtimestamp(mt)}) "
            f"-> solo std w cache: {acc:.4f}% ({ok}/{n})"
        )
    print()
    print("=" * 90)
    print("PRESETY (rozmiar ansambla)")
    print("=" * 90)
    print(f"{'Rozm.':<6} | {'Modele':<36} | {'Val %':<10} | Czas")
    print("-" * 90)
    for key in PRESET_SIZES:
        preset = ENSEMBLE_PRESETS[key]
        acc = run_evaluation(cache_data, corrected, preset["params"], preset["models"])
        comb = " + ".join(preset["models"])
        print(
            f"{key + 'M':<6} | {comb:<36} | {acc:.4f}%     | "
            f"{relative_time(preset['models']):.1f}x"
        )


def optimize_ensemble_weights(cache_data, corrected, comb_list, n_trials, seed=None):
    """Losowe wagi/TTA; seed=None bez resetu generatora."""
    if seed is not None:
        random.seed(seed)
    best_acc, best_params = 0.0, {}
    for _ in range(n_trials):
        weights = {m: random.uniform(-2.5, 2.5) for m in comb_list}
        params = {
            "model_weights": weights,
            "tta_bias": random.uniform(-1.5, 1.5),
            "tta_mode": random.choice(["none", "global", "selective"]),
            "tta_threshold": random.uniform(-2.5, -0.2),
            "use_dictionary": True,
        }
        acc = run_evaluation(cache_data, corrected, params, comb_list)
        if acc > best_acc:
            best_acc, best_params = acc, params
    return best_acc, best_params


def search_hunt(
    cache_path=CACHE_PATH,
    n_seeds=5,
    trials=TRIALS_HUNT,
    seeds=None,
    target_names=None,
    out_dir=RESULTS_DIR,
):
    """
    Wiele seedow dla wybranych zestawow (--target).
    Domyslnie: presety 1 i 2.
    """
    if not os.path.exists(cache_path):
        print(f"[!] Brak {cache_path}")
        return

    targets = resolve_hunt_targets(target_names)
    if not targets:
        print("[!] Brak poprawnych --target.")
        return

    seed_list = (seeds if seeds is not None else HUNT_SEEDS)[:n_seeds]
    cache_data = load_synced_cache(cache_path)
    train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
    corrected = build_corrected_cache(cache_data, train_gt)
    n = len(cache_data)

    print(f"[i] hunt: {n} val | {len(seed_list)} seedow | {trials} trials/zestaw")
    print(f"    seedy: {seed_list}")
    print(f"    cele:  {', '.join(t[0] for t in targets)}\n")

    best_global = {"acc": 0.0}
    report = {
        "timestamp": datetime.now().isoformat(),
        "val_samples": n,
        "trials_per_combo": trials,
        "seeds": seed_list,
        "targets": {},
        "best_global": None,
    }

    for key, label, models in targets:
        print(f"[Hunt] {label} ({' + '.join(models)}) -> Optymalizacja...")
        best_for_target = {"acc": 0.0, "seed": None, "params": {}}
        per_seed = []

        for seed in seed_list:
            acc, params = optimize_ensemble_weights(
                cache_data, corrected, models, trials, seed=seed
            )
            per_seed.append({"seed": seed, "acc": acc})
            if acc > best_for_target["acc"]:
                best_for_target = {"acc": acc, "seed": seed, "params": params}
            if acc > best_global["acc"]:
                best_global = {
                    "acc": acc,
                    "seed": seed,
                    "key": key,
                    "label": label,
                    "models": models,
                    "params": params,
                }

        print(
            f"       Najlepszy wynik: {best_for_target['acc']:.4f}% (seed {best_for_target['seed']})"
        )
        report["targets"][key] = {
            "label": label,
            "models": models,
            "per_seed": per_seed,
            "best": best_for_target,
        }

    report["best_global"] = best_global
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hunt_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Najlepszy wynik ogółem: {best_global['acc']:.4f}% dla {best_global['label']} (seed {best_global['seed']})")
    print(f"[+] Zapisano wyniki: {out_path}")
    apply_hunt_to_presets(report, cache_path=cache_path, apply=True, verbose=True)
    return report


def search_full(cache_path=CACHE_PATH, max_size=None, trials=TRIALS_PER_COMB):
    if not os.path.exists(cache_path):
        print(f"[!] Brak {cache_path}")
        return
    cache_data = load_synced_cache(cache_path)
    models = [
        m
        for m in MODEL_POOL
        if any(m in item.get("models", {}) for item in cache_data.values())
    ]
    missing = [m for m in MODEL_POOL if m not in models]
    if missing:
        print(f"[!] Brak w cache: {missing}")
        if "phase5" in missing:
            print("    Uruchom: python ocr.py cache --phase 5")
            return
    n = len(models)
    max_size = n if max_size is None else min(max_size, n)
    total_combos = sum(
        len(list(itertools.combinations(models, size)))
        for size in range(1, max_size + 1)
    )
    print(
        f"[i] search --full: {total_combos} kombinacji × {trials} trials "
        f"~ {total_combos * trials:,} ewaluacji"
    )
    if trials >= 500:
        print(
            "    Pełny benchmark może trwać wiele godzin (wygląda jak zawieszenie).\n"
            "    Szybciej: python ocr.py search --full --quick"
        )
    train_gt = os.path.join(DATA_DIR_DEFAULT, TRAIN_GT_REL)
    corrected = build_corrected_cache(cache_data, train_gt)

    def solo_std(m):
        c = sum(
            1
            for item in cache_data.values()
            if item["models"].get(m)
            and item["models"][m]["std"][0].lower() == item["label"].lower()
        )
        t = len(cache_data)
        return 100.0 * c / t, c, t

    lines = []
    report = {
        "timestamp": datetime.now().isoformat(),
        "val_samples": len(cache_data),
        "models": models,
        "solo_std": {},
        "solo_optimized": {},
        "by_size": {},
        "global_top": [],
    }

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out(f"BENCHMARK — {len(cache_data)} val, {n} modeli")
    out("=" * 90)
    random.seed(42)
    all_results = []

    for size in range(1, max_size + 1):
        combs = list(itertools.combinations(models, size))
        out(f"\n[Rozmiar {size}] {len(combs)} kombinacji...")
        size_entries = []
        for ci, comb in enumerate(combs, 1):
            comb_list = list(comb)
            acc, params = optimize_ensemble_weights(
                cache_data, corrected, comb_list, trials, seed=None
            )
            entry = {
                "size": size,
                "models": sort_models(comb_list),
                "acc": acc,
                "time_x": relative_time(comb_list),
                "params": params,
            }
            size_entries.append(entry)
            all_results.append(entry)
            if ci % max(1, len(combs) // 5) == 0 or ci == len(combs):
                print(f"    {ci}/{len(combs)}...", flush=True)
        size_entries.sort(key=lambda x: -x["acc"])
        report["by_size"][str(size)] = size_entries
        best = size_entries[0]
        out(f"  BEST: {' + '.join(best['models'])} => {best['acc']:.4f}%")

    all_results.sort(key=lambda x: -x["acc"])
    report["global_top"] = all_results[:15]
    out_path = os.path.join(RESULTS_DIR, "benchmark_results.txt")
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    out(f"\n[+] {out_path}\n[+] {json_path}")
    sync_ensemble_presets_from_benchmark(
        json_path=json_path, cache_path=cache_path, apply=True, verbose=True
    )


# --- update cache ---


def update_cache_phases(phases, cache_path=CACHE_PATH, data_dir=DATA_DIR_DEFAULT):
    val_gt = os.path.join(data_dir, VAL_GT_REL)
    if not os.path.exists(cache_path):
        print(f"[i] Brak cache: {cache_path} — tworzenie nowego słownika cache...")
        cache_data = {}
    else:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
    with open(val_gt, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if "\t" in line]
    valid_lines = [
        (p, lb) for p, lb in lines if os.path.exists(os.path.join(data_dir, p))
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Urządzenie: {device}")

    for phase in phases:
        spec = PHASE_CACHE.get(phase)
        if not spec:
            print(f"[!] Nieznana faza: {phase}")
            continue
        model_path = spec["model_path"]
        if not os.path.exists(model_path):
            print(f"[!] Brak modelu {phase}: {model_path}")
            continue
        old_snap = {}
        for img_path, _ in valid_lines:
            if img_path in cache_data:
                prev = cache_data[img_path].get("models", {}).get(phase)
                if prev:
                    old_snap[img_path] = prev["std"][0]
        if old_snap:
            acc_old, ok_old, n_old = _phase_solo_std_acc(
                {k: cache_data[k] for k in old_snap if k in cache_data}, phase
            )
            print(f"[i] {phase} przed: solo std {acc_old:.4f}% ({ok_old}/{n_old})")
        proc_path = model_path
        if not os.path.exists(os.path.join(proc_path, "tokenizer.json")):
            proc_path = spec["tokenizer_fallback"]
        processor = TrOCRProcessor.from_pretrained(proc_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        if device == "cuda":
            model = model.half()
        model.eval()
        bs = spec["batch_size"]
        beams = phase_num_beams(phase)
        nb = math.ceil(len(valid_lines) / bs)
        print(f"[+] {phase}: {len(valid_lines)} obrazów, batch={bs}, beams={beams}")
        with torch.no_grad():
            for i in tqdm(range(nb), desc=phase):
                batch = valid_lines[i * bs : (i + 1) * bs]
                images, images_tta = [], []
                for img_path, _ in batch:
                    img = Image.open(os.path.join(data_dir, img_path)).convert("RGB")
                    images.append(img)
                    images_tta.append(ImageEnhance.Contrast(img).enhance(1.2))
                pv = processor(images, return_tensors="pt").pixel_values.to(device)
                if device == "cuda":
                    pv = pv.half()
                out = model.generate(
                    pv,
                    num_beams=beams,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                preds = processor.batch_decode(out.sequences, skip_special_tokens=True)
                scores = out.sequences_scores.tolist()
                pv_tta = processor(images_tta, return_tensors="pt").pixel_values.to(
                    device
                )
                if device == "cuda":
                    pv_tta = pv_tta.half()
                out_tta = model.generate(
                    pv_tta,
                    num_beams=beams,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                preds_tta = processor.batch_decode(
                    out_tta.sequences, skip_special_tokens=True
                )
                scores_tta = out_tta.sequences_scores.tolist()
                for idx, (img_path, label) in enumerate(batch):
                    if img_path not in cache_data:
                        cache_data[img_path] = {"label": label, "models": {}}
                    cache_data[img_path]["label"] = label
                    for lk in spec.get("legacy_keys", ()):
                        cache_data[img_path]["models"].pop(lk, None)
                    cache_data[img_path]["models"][phase] = {
                        "std": (preds[idx].strip(), scores[idx]),
                        "tta": (preds_tta[idx].strip(), scores_tta[idx]),
                    }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        changed = 0
        for img_path, label in valid_lines:
            if img_path not in cache_data or phase not in cache_data[img_path].get(
                "models", {}
            ):
                continue
            new_p = cache_data[img_path]["models"][phase]["std"][0]
            if old_snap.get(img_path) != new_p:
                changed += 1
        acc_new, ok_new, n_new = _phase_solo_std_acc(cache_data, phase)
        print(
            f"[+] {phase}: zmieniono {changed}/{len(valid_lines)} pred std, "
            f"solo std {acc_new:.4f}% ({ok_new}/{n_new})"
        )
        if old_snap and changed == 0:
            print(
                "    [!] 0 zmian — sprawdź czy wgrałeś nowe wagi do "
                f"{model_path} (nie do kopii / checkpointu)"
            )

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"[+] Cache zapisany: {', '.join(phases)}")


# --- trening ALLROUND ---


def find_data_directory():
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "val_gt.txt" in files and (
                "train_gt.txt" in files
                or "train_gt_phase6c.txt" in files
                or "synthetic_gt.txt" in files
            ):
                print(f"[+] Kaggle data: {root}")
                return root
    return DATA_DIR_DEFAULT if os.path.exists(DATA_DIR_DEFAULT) else DATA_DIR_DEFAULT


def build_image_folder_lookup(data_dir):
    global IMAGE_FOLDER_LOOKUP
    IMAGE_FOLDER_LOOKUP = {}
    roots = []
    if os.path.exists("/kaggle/input"):
        roots.append("/kaggle/input")
    if os.path.exists(data_dir):
        roots.append(data_dir)
    for s_root in roots:
        for root, _, _ in os.walk(s_root):
            name = os.path.basename(root)
            if name in ("train", "val", "synthetic", "synthetic_phase6c"):
                IMAGE_FOLDER_LOOKUP.setdefault(name, [])
                if root not in IMAGE_FOLDER_LOOKUP[name]:
                    IMAGE_FOLDER_LOOKUP[name].append(root)


def detect_kaggle_weights(hints=()):
    if not os.path.exists("/kaggle/input"):
        return None
    candidates = []
    for root, _, files in os.walk("/kaggle/input"):
        if "config.json" not in files:
            continue
        if not ("model.safetensors" in files or "pytorch_model.bin" in files):
            continue
        if "easyocr_data" in root.replace("\\", "/"):
            continue
        candidates.append(root)
    for root in candidates:
        low = root.lower()
        if any(h in low for h in hints):
            print(f"[+] Kaggle wagi: {root}")
            return root
    if candidates:
        print(f"[+] Kaggle wagi: {candidates[0]}")
        return candidates[0]
    return None


def resolve_train_setup(size, warm_start=True):
    spec = TRAIN_SIZES[size]
    if warm_start:
        kw = detect_kaggle_weights(spec["weight_hints"])
        if kw:
            return kw, "warm", spec["out_warm"]
        if os.path.exists(spec["warm_path"]):
            return spec["warm_path"], "warm", spec["out_warm"]
        if spec["archive_warm"] and os.path.exists(spec["archive_warm"]):
            return spec["archive_warm"], "warm", spec["out_warm"]
        print("[!] Brak wag warm — fallback HF")
    return spec["hf_name"], "hf", spec["out_hf"]


def load_gt(path):
    rows = []
    if not os.path.exists(path):
        return pd.DataFrame(columns=["file_name", "text"])
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[1].strip():
                rows.append({"file_name": parts[0], "text": parts[1].strip()})
    return pd.DataFrame(rows)


def build_training_dataframe(data_dir, use_phase6c=True):
    real_df = load_gt(os.path.join(data_dir, "train_gt.txt"))
    synth_df = load_gt(os.path.join(data_dir, "synthetic_gt.txt"))
    diac_df = (
        load_gt(os.path.join(data_dir, "train_gt_phase6c.txt"))
        if use_phase6c
        else pd.DataFrame()
    )
    print(f"[i] real={len(real_df)} synth={len(synth_df)} diac={len(diac_df)}")
    parts = [df for df in (real_df, synth_df, diac_df) if len(df) > 0]
    if not parts:
        raise FileNotFoundError("Brak danych treningowych w easyocr_data")
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["file_name", "text"], keep="first"
    )
    if len(real_df) > 0 and len(synth_df) > 0:
        factor = min(8, max(1, round(len(synth_df) / len(real_df))))
        real_os = pd.concat([real_df] * factor, ignore_index=True)
        others = combined[~combined["file_name"].isin(real_df["file_name"])]
        combined = pd.concat([real_os, others], ignore_index=True)
    return combined.sample(frac=1.0, random_state=42).reset_index(drop=True)


class OCRDataset(Dataset):
    def __init__(self, df, root_dir, processor, transform=None, max_len=32):
        self.df = df
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _img_path(self, relative_path):
        parts = relative_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            sub, fn = parts[-2], parts[-1]
            for base in IMAGE_FOLDER_LOOKUP.get(sub, []):
                p = os.path.join(base, fn)
                if os.path.exists(p):
                    return p
        return os.path.join(self.root_dir, relative_path)

    def __getitem__(self, idx):
        fn = self.df.iloc[idx]["file_name"]
        text = self.df.iloc[idx]["text"]
        try:
            image = Image.open(self._img_path(fn)).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 96), color=(255, 255, 255))
        if self.transform:
            image = self.transform(image)
        pv = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_len, truncation=True
        ).input_ids
        labels = [
            x if x != self.processor.tokenizer.pad_token_id else -100 for x in labels
        ]
        return {"pixel_values": pv, "labels": torch.tensor(labels)}


TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomRotation(degrees=(-7, 7), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05),
        transforms.RandomApply(
            [transforms.ElasticTransform(alpha=12.0, sigma=5.0)], p=0.25
        ),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
            shear=(-4, 4),
            fill=255,
        ),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.35, fill=255),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.9))], p=0.2
        ),
    ]
)


def train_allround(
    size="base",
    warm_start=True,
    dry_run=False,
    epochs=4,
    resume=True,
):
    spec = TRAIN_SIZES[size]
    data_dir = find_data_directory()
    build_image_folder_lookup(data_dir)
    start, variant, out_dir = resolve_train_setup(size, warm_start)
    hf_name = spec["hf_name"]

    if os.path.exists("/kaggle"):
        n_gpu = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        if size == "large":
            bsz, gacc = 2, max(1, 16 // (2 * n_gpu))
        else:
            bsz, gacc = 8, 2
    else:
        bsz, gacc = (1, 16) if size == "large" else (4, 4)

    train_df = build_training_dataframe(data_dir)
    val_df = load_gt(os.path.join(data_dir, VAL_GT_REL))
    ep = epochs
    if dry_run:
        train_df = train_df.head(32 if size == "large" else 64)
        val_df = val_df.head(16 if size == "large" else 32)
        ep = 0.05 if size == "large" else 1

    from_hf = start.startswith("microsoft")
    try:
        processor = TrOCRProcessor.from_pretrained(start, local_files_only=not from_hf)
    except Exception:
        processor = TrOCRProcessor.from_pretrained(hf_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        start, local_files_only=not from_hf
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 32
    model.generation_config.num_beams = 1

    cer = evaluate.load("cer")

    def compute_metrics(pred):
        ps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        lids = pred.label_ids.copy()
        lids[lids == -100] = processor.tokenizer.pad_token_id
        ls = processor.batch_decode(lids, skip_special_tokens=True)
        return {"cer": cer.compute(predictions=ps, references=ls)}

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        gradient_accumulation_steps=gacc,
        eval_strategy="steps",
        predict_with_generate=True,
        num_train_epochs=ep,
        fp16=torch.cuda.is_available(),
        save_steps=200,
        eval_steps=200,
        logging_steps=20 if os.path.exists("/kaggle") else 50,
        learning_rate=2e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.08,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_checkpointing=spec["gradient_checkpointing"],
        ddp_find_unused_parameters=bool(os.path.exists("/kaggle")),
        disable_tqdm=bool(os.path.exists("/kaggle")),
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=OCRDataset(train_df, data_dir, processor, TRAIN_TRANSFORMS),
        eval_dataset=OCRDataset(val_df, data_dir, processor),
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=spec["early_stopping"])
        ],
    )
    print("=" * 70)
    print(f"ALLROUND {size.upper()} | {variant} | out={out_dir} | ep={ep}")
    print("=" * 70)
    ckpt = None
    if (
        resume
        and os.path.isdir(out_dir)
        and any(d.startswith("checkpoint-") for d in os.listdir(out_dir))
    ):
        ckpt = True
    trainer.train(resume_from_checkpoint=ckpt)
    final = os.path.join(out_dir, "final_model")
    trainer.save_model(final)
    processor.save_pretrained(final)
    print(f"[+] Zapisano: {final}")
    phase = "phase5" if size == "base" else "phase6"
    print(f"[i] Następnie: python ocr.py cache --phase {phase[-1]}")
