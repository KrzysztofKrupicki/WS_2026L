#!/usr/bin/env python3
"""
Jedna bramka CLI zamiast train_* / update_cache_* / search_*.

  python ocr.py search              # tryby produkcyjne (cache)
  python ocr.py search --full       # pełny benchmark → results/
  python ocr.py cache --phase all   # cache phase5 + phase6
  python ocr.py train --size base   # ALLROUND Base (phase5)
  python ocr.py train --size large  # ALLROUND Large (phase6)
  python ocr.py cer                 # CER kazdego modelu na val (z cache)
  python ocr.py analyze             # CSV + raport bledow

Kaggle 2× T4:
  torchrun --nproc_per_node=2 ocr.py train --size large
"""

import argparse

import ocr_lib as lib


def main():
    parser = argparse.ArgumentParser(
        description="TrOCR zad_3 — trening, cache, benchmark"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Benchmark ansamblu z cache")
    p_search.add_argument(
        "--full",
        action="store_true",
        help="Pełny przegląd kombinacji 1..N (domyślnie: tylko presety)",
    )
    p_search.add_argument(
        "--max-size", type=int, default=None, help="Limit rozmiaru ansambla"
    )
    p_search.add_argument(
        "--trials", type=int, default=None, help="Trials na kombinację (--full)"
    )
    p_search.add_argument(
        "--quick",
        action="store_true",
        help="Skrócony benchmark --full (150 trials/komb., ~minuty zamiast godzin)",
    )
    p_search.add_argument(
        "--hunt",
        action="store_true",
        help="Wiele seedow na presetach 1 i 2 (szukaj lepszych wag)",
    )
    p_search.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Liczba seedow przy --hunt (domyslnie 5)",
    )
    p_search.add_argument(
        "--target",
        nargs="+",
        choices=lib.HUNT_TARGET_CHOICES,
        metavar="TARGET",
        help=("Preset: 1 (solo 6), 2 (5+6), all (=2). Bez --target = oba presety"),
    )

    p_cache = sub.add_parser(
        "cache", help="Generuj predykcje do predictions_cache_full.pkl"
    )
    p_cache.add_argument(
        "--phase",
        nargs="+",
        required=True,
        choices=lib.CACHE_PHASE_CHOICES,
        help="5, 6 lub all (phase5 + phase6)",
    )

    p_train = sub.add_parser("train", help="Trening ALLROUND (Base lub Large)")
    p_train.add_argument("--size", choices=["base", "large"], default="base")
    p_train.add_argument(
        "--hf", action="store_true", help="Od zera (HF), bez warm-start"
    )
    p_train.add_argument("--dry-run", action="store_true", help="Krótki test pipeline")
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--no-resume", action="store_true")

    sub.add_parser("cer", help="CER na val per model (std/TTA) + ansambl")
    sub.add_parser("analyze", help="Analiza błędów val → CSV + raport bledow")

    args = parser.parse_args()

    if args.command == "search":
        if args.hunt:
            trials = args.trials if args.trials is not None else lib.TRIALS_HUNT
            lib.search_hunt(
                n_seeds=args.seeds,
                trials=trials,
                target_names=args.target,
            )
        elif args.full:
            if args.trials is not None:
                trials = args.trials
            elif args.quick:
                trials = lib.TRIALS_QUICK
            else:
                trials = lib.TRIALS_PER_COMB
            lib.search_full(max_size=args.max_size, trials=trials)
        else:
            lib.search_presets()
    elif args.command == "cache":
        if "all" in args.phase:
            phases = list(lib.MODEL_POOL)
        else:
            phases = [f"phase{p}" for p in args.phase if p != "all"]
        lib.update_cache_phases(phases)
    elif args.command == "cer":
        lib.report_val_cer()
    elif args.command == "analyze":
        lib.analyze_errors()
    elif args.command == "train":
        lib.train_allround(
            size=args.size,
            warm_start=not args.hf,
            dry_run=args.dry_run,
            epochs=args.epochs,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
