# Week 1

**Authors:**

Marcos Almansa, Arnau
Riubrogent Comas, Pol
Asbert Marcos, Gerard
Monserrat Llabrés, Pau

---

## Code structure

```
week1/
├── src/
│   ├── pipeline.py               # End-to-end detection pipeline
│   ├── video_source.py           # Video I/O abstraction
│   ├── evaluation.py             # COCO-style evaluation and annotation loading
│   ├── object_detection.py       # Connected-components bbox extraction (CarDetector, TemporalCarDetector)
│   ├── bbox_merging.py           # Bounding box merging strategies
│   ├── shadow_removal.py         # Shadow removal (HSV and chromaticity comparison)
│   ├── mask_postprocessing.py    # Morphological ops and small-blob filtering
│   ├── image_preprocessing.py   # Frame preprocessing (blur, etc.)
│   └── models/
│       ├── base.py               # BackgroundModel protocol
│       ├── still_models.py       # Static Gaussian and median background models
│       ├── adaptive_models.py    # Online-updating background models
│       └── opencv_models.py      # Wrappers for OpenCV MOG / MOG2 / LSBP
│
├── try_still_model.py            # Run and evaluate the static background model
├── try_pipeline.py               # Run and evaluate the full pipeline
├── compare_models.py             # Compare all models side by side
├── zbs_eval.py                   # Evaluate external ZBS pre-computed masks
│
├── alpha_study.py                # Sweep sensitivity threshold (alpha)
├── std_bias_study.py             # Sweep std bias parameter
├── adaptive_ro_study.py          # Sweep adaptive model rho (update rate)
├── adaptive_model_hyperparameter_search.py
├── still_model_hyperparameter_search.py
├── mog_model_hyperparameter_search.py
├── mog2_model_hyperparameter_search.py
├── lsbp_model_hyperparameter_search.py
├── sweep_temporal.py
│
├── plot_alpha_results.py
├── plot_adaptive_alpha_results.py
├── plot_std_bias_results.py
├── plot_mean_rho_results.py
└── plot_variance_rho_results.py
```
