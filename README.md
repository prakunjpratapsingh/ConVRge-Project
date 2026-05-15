# EEG_Preprocessing_and_ModelTraining.ipynb

# EEG Processing Pipeline (Multimodal-Fusion-Ready)

This repository contains an EEG preprocessing, feature extraction, and model training pipeline designed for **multimodal fusion with ECG and eye tracking**.
The implementation ensures **strict temporal alignment**, **reproducibility**, and **consistent feature generation** across modalities.

---

## Objective

- Extract **spectral, temporal, and statistical features** from continuous EEG recordings
- Align EEG windows with ECG and eye-tracker processing
- Generate **window-level and trial-level features** for cross-domain classification and multimodal fusion

---

## Core EEG Processing Pipeline

The continuous EEG signal undergoes the following preprocessing stages before windowing:

- 8-channel acquisition from the Unicorn Hybrid Black headset at 250 Hz
- ADC saturation rejection (`±740,000` µV) with linear interpolation
- FIR bandpass filter (**1–40 Hz**, Hamming window) to retain cognitive frequency bands
- Notch filter at **50 Hz + 100 Hz** to remove power-line interference and harmonics
- Common Average Reference (CAR) to reduce common-mode noise
- Bad-channel detection (>3× median std) with spherical interpolation
- Independent Component Analysis (FastICA) with frontal Fz used as an EOG proxy for ocular artifact removal
- Whole-session per-channel Z-score normalization with ±5σ clipping

**No subject-specific tuning is applied** — the same parameters are used across all participants and both environments (VR and RW) to enable cross-domain comparison.

---

## Feature Extraction

From each window, **14 features per channel × 8 channels = 112 features** are extracted:

### Frequency-Domain Features

- Relative band powers (normalized by total power and log-transformed):
  - **delta** (1–4 Hz)
  - **theta** (4–8 Hz)
  - **alpha** (8–13 Hz)
  - **beta** (13–30 Hz)
  - **gamma** (30–40 Hz)
- **theta/alpha ratio** — engagement index, the most replicated workload marker

### Hjorth Parameters

- **Activity** — variance (amplitude)
- **Mobility** — mean frequency
- **Complexity** — bandwidth

### Statistical Moments

- mean, standard deviation, variance, skewness, kurtosis

---

## Feature Selection

A **consensus-voting approach** combining three independent methods is used:

- ANOVA F-test
- Mutual Information
- Recursive Feature Elimination with ExtraTrees

A feature is retained if it is selected by **at least 2 of the 3 methods**. This yields **21 consensus features**, with four selected unanimously (3/3): `Fz_std`, `Fz_delta`, `PO7_gamma`, `PO7_mobility`.

---

## Windowing Strategy (Aligned with ECG and Eye Tracking)

| Parameter      | Value                                          |
|----------------|------------------------------------------------|
| Window length  | 2 s / 5 s / 10 s (multi-resolution analysis)   |
| Overlap        | 50%                                            |
| Step size      | `WINDOW_SEC / 2`                               |
| Start offset   | 0 seconds                                      |

For the multimodal-fusion pipeline, **5 s windows** are used to match ECG and eye-tracker windowing.

---

## Global Window Count (N)

The number of windows per trial (**N**) is computed from the **shortest valid trial duration across all participants and both environments (VR + RW)**, so that every trial contributes the same number of samples and the model is not biased by trial duration.

### Procedure

1. Compute effective duration for every valid trial.
2. Find the **global minimum duration**:

```
   min_duration = min(all_trial_durations)
```

3. Compute N using:

```
   N = floor((min_duration − W) / S) + 1
```

   where W is window length and S is step size.

### Result for 5 s Windows

- Shortest valid trial: 61.3 s
- N = 23 windows per trial
- Effective duration covered: 60 s

---

## Deterministic Window Selection (Cross-Modality Alignment)

To ensure that EEG, ECG, and eye-tracker pipelines select the **same temporal segments** of each trial, a stable hashing function seeds the random window selection.

### Previous Approach

- Global RNG (`seed = 42`)
- Window selection depended on **processing order**
- Not guaranteed to align across modalities (EEG vs ECG vs eye tracking)

### Updated Approach

```python
seed = stable_hash(pid, trial_id, world, WINDOW_SEC, MASTER_SEED)
rng  = np.random.RandomState(seed)
```

This guarantees:

- Identical window indices selected across all three modalities for the same trial
- Full reproducibility — the same windows are chosen on every run
- The model is not biased by trial timing or processing order

---

## Trial-Level Representation

For trial-level export (used in fusion), features are aggregated across the N selected windows:

- The **mean** of each feature across windows is computed
- Two metadata fields are included:
  - **n_valid_windows** — number of windows used (fixed at N = 23 for 5 s windows)
  - **used_duration** — effective duration covered by the windows

Because of the 50% overlap, the used duration is:

```
used_duration = (N − 1) × step + window = 60 s
```

---

## Output Format

Each trial is stored as a single row in the output CSV file, with the following structure:

- world (VR or RW)
- participant_id
- trial_id
- level and trial identifiers
- n_valid_windows
- used_duration
- 21 EEG feature columns (consensus-selected)

Separate CSV files are generated for each participant and each world.

---

## Model Training and Evaluation

The pipeline supports multiple classification tasks at the window level:

### Binary Tasks

- **VR vs Real** — environment classification
- **Image Whole Duration vs Image 30s** — memory load
- **8 vs 12 Objects** — object load

### Scenario Comparisons

- L1 vs L2 (object load, image visible)
- L3 vs L4 (object load, image hidden)
- L1 vs L3 (memory load, 8 objects)
- L2 vs L4 (memory load, 12 objects)

Each scenario evaluated in three settings: Combined, Real only, VR only.

### 4-Class Classification

- L1 vs L2 vs L3 vs L4

### Cross-Domain Transfer

- Train VR → Test Real
- Train Real → Test VR

### Validation

- **Leave-One-Participant-Out (LOPO)** cross-validation across 6 participants
- Three classifiers compared per task: LinearSVM, ExtraTrees, RandomForest
- Reported metrics: balanced accuracy (BAC) and macro F1-score

---

## Explainability (XAI)

Feature ranking heatmaps are generated for every classification task:

- Whole-dataset heatmap (single row, all participants pooled)
- Per-participant heatmap showing inter-subject variability
- Cross-domain heatmap (Train VR → Test Real and reverse) using permutation importance

These reveal task-specific cortical signatures:

- **Environment (VR vs Real):** Frontal dominance (`Fz_std`, `Fz_var`, `Fz_delta`)
- **Memory recall:** Parietal region (`Pz_activity`)
- **Visual load (image visible):** Parietal region (`Pz_delta`)
- **Visual load (image hidden):** Occipital region (`PO7_mobility`)

---

## EEG–Performance Correlation and Duration-Independence Check

The pipeline validates that the model captures genuine cognitive workload rather than trial duration:

- **Workload–performance correlation:** ρ = −0.418, p < 0.0001, n = 141 trials
  - VR: ρ = −0.308
  - Real: ρ = −0.467
- **Prediction–duration correlation:** ρ = +0.166 (near-zero — duration confound eliminated)

A three-panel figure visualizes the difficulty chain: task performance, EEG-predicted P(High Workload), and the correlation scatter.

---

## Role in Multimodal Fusion

For correct fusion with ECG and eye tracking, the following must remain **identical across all pipelines**:

- Trial boundaries
- Window size and overlap
- Number of windows per trial (N)
- Window selection strategy (deterministic seeding via `stable_hash`)
- Trial-level aggregation method

Maintaining this consistency ensures that rows from different modalities correspond to the **same temporal segments**, enabling reliable late fusion via majority voting and Logistic Regression stacking.




# Sensor Processing Pipeline for Multimodal Fusion

Follow preprocessing and windowing strategy as mentioned below for all modalities such as EEG, ECG and eye tracking to have the same temporal structure so that the data can be aligned correctly during fusion.

---

## Trial Extraction

The continuous recordings are first segmented into trials using event markers (`trial_start` and `trial_end`). Each trial corresponds to a specific condition (e.g., L01T01) and is associated with a participant and a recording environment (VR or real-world).

---

## Windowing Strategy

To capture temporal dynamics while maintaining consistency across trials, the EEG signal within each trial is divided into overlapping windows.

A window length of 5 seconds is used with a 50% overlap. This results in a step size of 2.5 seconds between consecutive windows. For example : EEG sampled at 250 Hz, this corresponds to:


The windows therefore follow a sliding pattern such as:
0–5 s, 2.5–7.5 s, 5–10 s, and so on.

---

## Number of Windows per Trial

Since trials have different durations using all possible windows would introduce variability in the number of samples per trial. This can bias the model as longer trials would contribute more training data and potentially dominate the learning process. To avoid this, a fixed number of windows is used for every trial.

This number is determined based on the shortest valid trial in the dataset:

N = floor((T_min − W) / S) + 1

where:

* T_min is the shortest trial duration
* W is the window length (5 s)
* S is the step size (2.5 s)

example : the shortest trial is approximately 61.3 seconds, which results in:

N = 23 windows per trial

---

## Window Selection and Alignment

For each trial, all possible overlapping windows are first generated. From these, exactly N windows are selected.

The selection is random but deterministic. A fixed hashing function based on participant ID, trial ID, world, and a global seed is used to initialise the random generator. This ensures that the same windows are selected every time the pipeline is run.

More importantly, this guarantees that all modalities (EEG, ECG, eye tracking) select the same temporal segments, which is essential for correct multimodal fusion.

---


## Trial-Level Representation

The final representation of each trial is obtained by aggregating features across its windows. Specifically, the mean value of each feature over the selected windows is computed.

In addition, two metadata fields are included:

* **n_valid_windows**: number of windows used (example fixed at N = 23)
* **used_duration**: effective duration covered by the windows

Because of the overlap, the used duration is not simply N × window size. Instead, it is computed as:

used_duration = (N − 1) × step + window

For the chosen parameters, this corresponds to approximately 60 seconds.

---

## Output Format

Each trial is stored as a single row in the output CSV file, with the following structure:

* world (VR or RW)
* participant_id
* trial_id
* level and trial identifiers
* n_valid_windows
* used_duration
* EEG feature columns

Separate CSV files are generated for each participant and each world.

---

## Role in Multimodal Fusion

feature extraction differs between EEG, ECG, and eye tracking, the following aspects must remain identical across all pipelines:

* Trial boundaries
* Window size and overlap
* Number of windows per trial (N)
* Window selection strategy
* Trial-level aggregation

Maintaining this consistency ensures that rows from different modalities correspond to the same temporal segments, enabling reliable late fusion.

---

## ECG_Export_for_Fusion.ipynb 

#  ECG Processing Pipeline (EEG-Aligned)

This repository contains an ECG feature extraction pipeline designed for **multimodal fusion with EEG**.  
The implementation ensures **strict temporal alignment**, **reproducibility**, and **consistent feature generation** across modalities.

---

##  Objective

- Extract **Heart Rate Variability (HRV)** features from ECG signals  
- Align ECG windows with EEG processing  
- Generate **trial-level features** for multimodal classification  

---

## Core ECG Processing (UNCHANGED)

The physiological signal processing pipeline remains **identical to the original implementation**:

- ECG channel extraction (`ECGBIT0`)
- Bandpass filtering (**0.5–40 Hz**, Butterworth)
- R-peak detection using `neurokit2`
- RR interval computation
- Artifact correction:
  - Remove RR < 0.3s or > 2.0s
  - Interpolate missing values
- HRV feature extraction via `compute_rr_features`

**No changes were made to core ECG preprocessing or physiological feature computation.**

---

##  Feature Adjustments

### Removed Features

- `ECG_sampen`
- `ECG_apen`

### Reason

- These features frequently produced **NaN values**
- Caused instability during aggregation and modeling
- Removed to ensure robustness

---

## Windowing Strategy (Aligned with EEG)

| Parameter        | Value                          |
|----------------|-------------------------------|
| Window length   |  5s (matches EEG)   |
| Overlap         | 50%                           |
| Step size       | `WINDOW_SEC / 2`              |
| Start offset    | 0 seconds (EEG-aligned)   |

---

### Critical Alignment Fix

**Removed:**
SKIP_SEC = 30

## Global Window Count (N) 

The number of windows per trial (**N**) is computed from the **shortest valid trial duration across all participants and worlds (VR + RW) gives have uniform Window so model not be biased due to duration of the trial **.

### Procedure
1. Compute effective duration for every valid trial.
2. Find the **global minimum duration**:

 min_duration = min(all_trial_durations)

##  Deterministic Window Selection (So Model not be biased on timing of the trials and the seed =49 and stable hash guves same random window among all sensors)

### Previous Approach
- Global RNG (`seed = 42`)
- Window selection depended on **processing order**
- Not guaranteed to align across modalities (EEG vs ECG)

### Updated Approach

seed = stable_hash(pid, trial_id, world, WINDOW_SEC, MASTER_SEED)
rng = np.random.RandomState(seed)


