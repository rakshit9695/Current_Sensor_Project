# ⚡ Industrial Current Analyser — Dual-Machine RAG + ML Pipeline

> **Live App → [rakshit9695-current-sensor-project-app-qinted.streamlit.app](https://rakshit9695-current-sensor-project-app-qinted.streamlit.app/)**

A Streamlit web application that reads 3-phase electrical current data from industrial equipment, validates signal quality, classifies operating state using deterministic rules (RED / AMBER / GREEN), and provides an optional deep-learning layer (iTransformer) for cycle-aware prediction — all in real time, with every decision transparent, explainable, and fully configurable.

**Two machines. Same pipeline. One app.**

| Machine | What it is | Current Range | States |
|---|---|---|---|
| 🔥 **Wave Soldering Furnace** | High-power thermal process equipment | 0.2 – 24 A | Idle · No Load · Load |
| 🏭 **MV Conveyer** | Motor-driven conveyor belt | 0.1 – 0.8 A | HALT · IDLE · RUNNING |

---

## Table of Contents

1. [The Core Idea — What, Why and How](#1-the-core-idea--what-why-and-how)
2. [The Physical System](#2-the-physical-system)
3. [The Data](#3-the-data)
4. [Architecture Overview](#4-architecture-overview)
5. [Pipeline Deep-Dive — Module by Module](#5-pipeline-deep-dive--module-by-module)
   - [5.1 Data Loader](#51-data-loader)
   - [5.2 Windowing](#52-windowing)
   - [5.3 FFT Analysis](#53-fft-analysis)
   - [5.4 Validation](#54-validation)
   - [5.5 Feature Engineering](#55-feature-engineering)
   - [5.6 RAG Classifier](#56-rag-classifier)
   - [5.7 State Manager](#57-state-manager)
6. [The ML Layer — iTransformer](#6-the-ml-layer--itransformer)
   - [6.1 Why Add ML When Rules Work?](#61-why-add-ml-when-rules-work)
   - [6.2 Auto-Label Generation](#62-auto-label-generation)
   - [6.3 iTransformer Architecture](#63-itransformer-architecture)
   - [6.4 Multi-Task Training](#64-multi-task-training)
   - [6.5 Inference](#65-inference)
7. [The UI — Three Pages](#7-the-ui--three-pages)
8. [Machine-Specific Differences](#8-machine-specific-differences)
9. [Every Threshold, Explained](#9-every-threshold-explained)
10. [Frequently Asked Questions](#10-frequently-asked-questions)
11. [How to Run](#11-how-to-run)
12. [File Reference](#12-file-reference)

---

## 1. The Core Idea — What, Why and How

### What

Every industrial machine draws a characteristic pattern of electrical current. A furnace heating up draws high sustained current; a conveyor cycling on and off draws short repetitive bursts. This application reads those current patterns and answers three questions:

1. **Is the data trustworthy?** — validation layer
2. **What is the machine doing right now?** — rule-based RAG classification
3. **Can we predict what it will do next?** — optional ML (iTransformer) layer

### Why

On a factory floor, operators need to know — at a glance — whether a machine is idle, warming up, or in full production. Historically this requires either a human watching a panel or expensive SCADA integration. This tool achieves the same thing using only a current sensor (a clamp meter on the power cable) and a CSV export. It costs nothing beyond the sensor hardware.

**Why rules over ML?** Because a factory technician needs to understand *why* the system says "GREEN — Load". With rules, the answer is always: "because RMS = 14.2 A, which is above the 12.0 A threshold you set". There is no hidden weight matrix. No training data dependency. No model drift. The ML layer is offered as an *optional enhancement* — not a replacement — for users who want cycle-position tracking and job-type clustering.

### How

A 3-phase current clamp sensor samples RMS current once per second (1 Hz). The readings are exported to CSV. The application loads that CSV, slices it into overlapping time windows, extracts time-domain and frequency-domain features from each window, runs four quality checks, applies deterministic threshold rules to assign RED / AMBER / GREEN, and smooths the output with hysteresis. The entire pipeline is deterministic — the same input always produces the same output.

---

## 2. The Physical System

### What is a 3-phase AC current sensor measuring?

Industrial equipment in most countries runs on **3-phase alternating current (AC)** at 50 Hz (or 60 Hz). Three separate conductors each carry a sinusoidal current waveform offset by 120°. The sensor wraps around all three conductors and reports:

- **I1, I2, I3** — the RMS current on each phase (Amperes)
- **Frequency** — the measured grid frequency (should be 50.00 Hz)

### What is RMS and why does it matter?

RMS stands for **Root Mean Square**. Raw AC current swings positive and negative 50 times per second. The RMS value is the equivalent DC current that would deliver the same power. It is the standard industrial measurement of "how much current is flowing".

**Crucially, the sensor has already computed RMS for us.** Each row in the CSV is one RMS value per second — not a raw waveform sample. This means:

- We **cannot** do waveform-level analysis (e.g., detect individual 50 Hz cycles)
- We **can** detect operational patterns (heater cycling, motor on/off) that happen over seconds
- The Nyquist frequency is 0.5 Hz (half the 1 Hz sample rate), so FFT analysis reveals patterns up to one event every 2 seconds

### Physical significance of the three states

| State | Physical reality | Current signature | Why it matters |
|---|---|---|---|
| 🔴 **RED — Idle / HALT** | Machine powered off or in standby. No heating elements or motors active. Only control electronics drawing negligible current. | I_avg < 2.0 A (furnace) or < 0.10 A (conveyer) | Knowing idle time helps calculate utilisation rates and energy waste. |
| 🟡 **AMBER — No Load / IDLE** | Machine powered and active (heaters cycling, motor energised) but not processing product. The furnace maintains temperature; the conveyor is energised but not moving material. | Moderate current with regular cycling pattern. | Excessive AMBER time means the machine is burning energy without producing output — a cost optimisation opportunity. |
| 🟢 **GREEN — Load / RUNNING** | Full production. Furnace processing PCBs through the solder wave; conveyor transporting product. Maximum current draw with sustained high power. | I_avg > 12.0 A (furnace) or > 0.35 A (conveyer) | The productive state. Operators want to maximise GREEN time. |

### Why current is a reliable proxy for machine state

Ohm's law: **P = I × V**. At a fixed supply voltage (which 3-phase mains effectively is), current is directly proportional to power. More power = more work being done. The current signature is therefore a direct, physical measurement of how hard the machine is working. It cannot be faked, is difficult to misconfigure, and requires no integration with the machine's own control system.

---

## 3. The Data

### Wave Soldering Furnace

| Property | Value |
|---|---|
| File | `data/Current Data Final - Soldering Furnace - Sheet1.csv` |
| Duration | ~50 hours (~179,000 samples) |
| Sample rate | 1 Hz (one reading per second) |
| Timestamp format | `MM:SS.f` — wraps every 60 minutes |
| I_avg range | 0.2 – 24 A |
| Contains | Active production, no-load warm-up, and a full shutdown |

| Column | Physical meaning |
|---|---|
| `TimeStamp` | Time of measurement in MM:SS format. **Wraps back to 00:00.0 every hour.** |
| `I1` | RMS current on Phase 1 (Amperes) |
| `I2` | RMS current on Phase 2 (Amperes) |
| `I3` | RMS current on Phase 3 (Amperes) |
| `Frequency` | Sensor-reported AC grid frequency (should be ~50 Hz) |

> **Why does the timestamp wrap?** The sensor's internal clock counts minutes and seconds within the current hour, then resets. Over 50 hours, this creates dozens of discontinuities. The data loader must detect each wrap and reconstruct a monotonically increasing timeline — this is the first and most critical data-cleaning step.

### MV Conveyer

| Property | Value |
|---|---|
| Files | `Stop_and_Run_MV_Conveyer.csv` + `Cyclic_Run_MV_Conveyer.csv` |
| Timestamp format | Standard ISO datetime (no wrapping) |
| I_avg range | 0.10 – 0.79 A |
| Cycle period | ~15 seconds (11 s ON + 4 s OFF) |
| Contains | Continuous run (stop-and-run) and periodic cycling data |

> **Why two files?** The stop-and-run file captures a simple on/off/on sequence (provides lookback context for ML training). The cyclic file captures the conveyor's normal operating pattern — periodic short bursts — which is what the RAG page visualises.

---

## 4. Architecture Overview

```
                            ┌─────────────────────────────────┐
                            │          CSV File(s)            │
                            └──────────────┬──────────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────────────┐
                            │      data_loader.py              │
                            │  Parse timestamps, normalise     │
                            │  column names, compute i_avg     │
                            └──────────────┬───────────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────────────┐
                            │      windowing.py                │
                            │  Slice into overlapping windows  │
                            │  (60s furnace / 3s conveyer)     │
                            └──────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┼────────────────────────┐
                    │                      │                        │
                    ▼                      ▼                        ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────────┐
        │  fft_analysis.py  │  │  validation.py    │  │ feature_engineering.py │
        │  FFT spectrum     │  │  4 quality checks │  │  20+ scalar features  │
        │  per window       │  │  per window       │  │  per window           │
        └────────┬──────────┘  └────────┬──────────┘  └──────────┬─────────────┘
                 │                      │                        │
                 └──────────────────────┼────────────────────────┘
                                        │
                                        ▼
                            ┌──────────────────────────────────┐
                            │   rag_classifier.py              │
                            │   Deterministic rule-based        │
                            │   RED / AMBER / GREEN             │
                            └──────────────┬───────────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────────────┐
                            │   state_manager.py               │
                            │   Hysteresis smoothing            │
                            │   (3-window minimum hold)         │
                            └──────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                              │
                    ▼                                              ▼
        ┌───────────────────┐                         ┌───────────────────────┐
        │   app.py          │                         │   ml/ (optional)      │
        │   Streamlit UI    │                         │   iTransformer model  │
        │   3 pages         │                         │   Auto-labelling      │
        └───────────────────┘                         │   Multi-task training │
                                                      └───────────────────────┘
```

Both machines share the exact same pipeline structure. The only differences are threshold values, timestamp parsing, and state labels. This symmetry is deliberate — adding a third machine means duplicating the folder, adjusting thresholds, and registering it in `MACHINE_META` inside `app.py`.

---

## 5. Pipeline Deep-Dive — Module by Module

### 5.1 Data Loader

**Files:** `wave_soldering/data_loader.py` · `mv_conveyer/data_loader.py`

**Problem it solves:** Raw sensor data is messy. Timestamps wrap, column names vary, and current values might be negative (sensor artefact). The loader produces a clean, standardised DataFrame that every downstream module can rely on.

**Wave Soldering — timestamp reconstruction:**

The furnace sensor records timestamps as `MM:SS.f` (e.g., `32:45.1`). After 59:59.9, it wraps back to 00:00.0. Over 50 hours, this happens ~50 times. The algorithm:

1. Parse each `MM:SS.f` string into total seconds (e.g., `32:45.1` → 1965.1 s)
2. Scan for drops greater than 30 minutes — each drop marks an hour boundary
3. Add a cumulative hour offset at each boundary
4. Convert the reconstructed seconds into `pd.Timestamp` objects

This produces a monotonically increasing timeline that all time-based operations (windowing, plotting, playback) depend on.

**MV Conveyer — multi-file concatenation:**

The conveyer uses standard ISO datetimes (no wrapping). The loader accepts a single file, a list of files, or a directory. When given multiple files, it loads each one independently, concatenates them, and sorts by timestamp. This allows combining the stop-and-run file (lookback context) with the cyclic file (operational data) in a single DataFrame.

**Output schema (both machines):**

| Column | Type | Description |
|---|---|---|
| `timestamp` | `pd.Timestamp` | Continuous, monotonically increasing |
| `timestamp_str` | `str` | Original display format |
| `i1`, `i2`, `i3` | `float` | Phase RMS currents (A), clipped to ≥ 0 |
| `i_avg` | `float` | `mean(i1, i2, i3)` — computed, not from source |
| `frequency` | `float` | Grid frequency (Hz), defaults to 50.0 if missing |

> **Why compute `i_avg` ourselves?** The source CSV may or may not contain an average column, and its name varies. Computing it fresh from the three phases guarantees consistency and eliminates a class of data-quality bugs.

---

### 5.2 Windowing

**Files:** `wave_soldering/windowing.py` · `mv_conveyer/windowing.py`

**Why windows, not individual samples?**

A single 1-second current reading tells you almost nothing. Is 13 A high? It depends on context. But if you look at 60 consecutive readings and see they are *all* above 12 A with low variance, you can confidently say the furnace is under load. Windowing converts a stream of noisy point measurements into a sequence of statistically meaningful summaries.

**How it works:**

1. Start at the first timestamp in the DataFrame
2. Collect all rows within `[t, t + window_size_sec)`
3. Step forward by `window_size_sec × (1 - overlap)` seconds
4. Repeat until the end of the dataset
5. Discard any trailing window with fewer than 50% of the expected samples

**Key parameters:**

| Parameter | Furnace default | Conveyer default | Physical reasoning |
|---|---|---|---|
| `window_size_sec` | 60 s | 3 s | Furnace states last minutes; conveyer cycles every ~15 s (3 s fits inside a single IDLE gap) |
| `overlap` | 0.50 (50%) | 0.50 (50%) | Ensures state transitions are captured at half-window resolution |

**Physical significance of overlap:**

Without overlap, a state transition happening in the middle of a window gets averaged away — the window looks like a blend of two states. With 50% overlap, the step size is 30 s (furnace), so a transition is detected within ±30 s of when it actually occurred. This is the classic trade-off between temporal resolution and noise reduction.

**Data structure:** Each window is a `Window` dataclass containing `index`, `start_time`, `end_time`, `data` (DataFrame slice), and row indices. This keeps provenance intact — you can always trace a classification back to the exact rows that produced it.

---

### 5.3 FFT Analysis

**Files:** `wave_soldering/fft_analysis.py` · `mv_conveyer/fft_analysis.py`

**What FFT does in plain English:**

Imagine you record the hum of a machine for 60 seconds. FFT tells you: "There is a strong component repeating 4 times per second, a weaker one at 8 times per second, and some random noise." It decomposes a signal into its constituent frequencies.

**The critical adaptation for 1 Hz RMS data:**

The sensor samples at 1 Hz. By the Nyquist theorem, the highest frequency we can detect is 0.5 Hz (one event every 2 seconds). We **cannot** see the 50 Hz AC line frequency in the FFT — that would require ≥100 Hz sampling. This is not a limitation; it is by design. What we *can* see is the machine's **operational cycling rhythm**:

- **Furnace heaters** switch on and off roughly every 4–5 seconds → shows up as a peak at ~0.2–0.25 Hz
- **Conveyer motor** cycles on for 11 s and off for 4 s → shows up at ~0.067 Hz (~67 mHz)

These operational frequencies are *exactly* what distinguishes an active machine from an idle one.

**Algorithm per window:**

1. Extract the `i_avg` column (and each phase individually)
2. Apply a **Hann window** to the signal (reduces spectral leakage — without it, sharp transitions at window edges create fake frequency peaks)
3. Compute the one-sided FFT with amplitude scaling (`2/n`)
4. Identify the **dominant frequency** (largest magnitude, excluding DC)
5. Search for the **2nd and 3rd harmonics** at 2× and 3× the dominant frequency (±0.05 Hz tolerance)
6. Compute **THD** (Total Harmonic Distortion) = `√(H2² + H3²) / fundamental`
7. Compute **total spectral energy** and **high-frequency energy ratio** (energy above `sample_rate / 4`)

**Output:** An `FFTResult` dataclass per channel containing `freqs`, `magnitudes`, `fundamental_freq`, `fundamental_mag`, `harmonic2_mag`, `harmonic3_mag`, `total_energy`, `high_freq_energy`, and `thd`.

> **Why Hann window?** A rectangular window (no windowing) assumes the signal repeats perfectly at the boundaries. Real data doesn't. The Hann window tapers the edges to zero, preventing spectral leakage — false energy spread across frequencies that would corrupt our dominant frequency detection.

> **Why harmonics?** Harmonics indicate non-sinusoidal operational patterns. A furnace under load draws current in a more complex pattern than a furnace idling. High THD correlates with active production.

---

### 5.4 Validation

**Files:** `wave_soldering/validation.py` · `mv_conveyer/validation.py`

**Why validate at all?**

Garbage in, garbage out. If the sensor was disconnected, the grid was unstable, or a wiring fault zeroed out one phase, the classification would be meaningless. Validation is the gate that prevents bad data from reaching the classifier.

**Four checks, each with a physical reason:**

#### Check 1 — Grid Frequency

| | Detail |
|---|---|
| **What:** | The `frequency` column should read 50 Hz ±0.5 Hz in ≥90% of samples in the window |
| **Why:** | If the grid frequency is wrong, the sensor may have lost phase lock. Readings taken during phase-lock loss are unreliable. |
| **Physical significance:** | Grid frequency is maintained by the power utility at 50.00 Hz (±0.02 Hz in normal operation). A reading of 48 Hz or 52 Hz almost certainly means sensor malfunction, not an actual grid event. |

#### Check 2 — Current Range

| | Detail |
|---|---|
| **What:** | All current values must be finite and non-negative. At least one phase must exceed 0.01 A. |
| **Why:** | Negative current from an RMS sensor is physically impossible (RMS is always ≥ 0). NaN/Inf values indicate data corruption. |
| **Physical significance:** | An RMS reading of exactly 0.000 A across all three phases means the sensor is disconnected or the machine's breaker is open. |

#### Check 3 — Phase Consistency

| | Detail |
|---|---|
| **What:** | The coefficient of variation (CV = σ/μ) across the three phases must be below a threshold. |
| **Why:** | In a properly wired 3-phase system under load, all three phases should carry similar current. A CV > 0.80 (furnace) or > 1.50 (conveyer) suggests one phase is dead — likely a sensor wiring issue. |
| **Physical significance:** | Phase imbalance exists in all real systems (no motor or heater is perfectly balanced). But *extreme* imbalance indicates a fault. The conveyer threshold is higher (1.50) because during idle periods, motor Phase 2 legitimately drops near zero while Phase 1 remains energised by auxiliary circuitry. |

#### Check 4 — Energy Floor

| | Detail |
|---|---|
| **What:** | `mean(i_avg)` must exceed 0.05 A. |
| **Why:** | Below this threshold, readings are dominated by sensor noise and quantisation error. The signal-to-noise ratio is too low for meaningful classification. |
| **Physical significance:** | 0.05 A at 400 V is 20 W — less than a light bulb. Any reading this low is indistinguishable from measurement noise. |

**Dataset-level validation:** If fewer than 80% of windows pass all four checks, the entire dataset is flagged INVALID. The RAG page is blocked. This prevents operators from acting on unreliable analysis.

---

### 5.5 Feature Engineering

**Files:** `wave_soldering/feature_engineering.py` · `mv_conveyer/feature_engineering.py`

This module distils a window of 60 raw readings (furnace) or 3 readings (conveyer) into **20+ scalar features** that compactly describe the machine's behaviour.

**Time-domain features (computed directly from current values):**

| Feature | Formula | Physical meaning |
|---|---|---|
| `rms_i1`, `rms_i2`, `rms_i3` | `√(mean(x²))` per phase | Effective current per phase — how much power each phase delivers |
| `rms_i_avg` | `√(mean(i_avg²))` | Overall power draw — the single most important classification feature |
| `variance_i_avg` | `var(i_avg)` | Current fluctuation within the window. High variance = heaters actively cycling. Low variance = stable state (either full-on or full-off). |
| `peak_to_peak` | `max(i_avg) - min(i_avg)` | Total swing. Large swings indicate transitions or cycling. |
| `phase_imbalance` | `std(rms_phases) / mean(rms_phases)` | How unevenly the load is distributed. Healthy 3-phase loads have imbalance < 0.2. |

**Frequency-domain features (from FFT):**

| Feature | Physical meaning |
|---|---|
| `fundamental_freq` | The dominant operational cycling rate (e.g., 0.25 Hz = heaters cycling every 4 s) |
| `fundamental_energy` | Strength of the dominant cycle — stronger = more regular pattern |
| `harmonic2_energy`, `harmonic3_energy` | Energy at 2× and 3× the fundamental — indicates non-sinusoidal operational patterns |
| `total_spectral_energy` | Total signal energy in frequency domain — correlates with overall activity level |
| `high_freq_ratio` | Fraction of energy above 0.25 Hz — high values may indicate vibration or transients |

**Derived features:**

| Feature | Formula | Physical meaning |
|---|---|---|
| `thd` | `√(H2² + H3²) / fundamental` | Total Harmonic Distortion proxy. High THD means the operational pattern is complex (multiple overlapping cycles), which is characteristic of active production. |

> **Why this specific set?** Each feature was chosen because it maps to a physical phenomenon. RMS measures power. Variance measures stability. THD measures complexity. Phase imbalance measures health. Together, they form a fingerprint of the machine's operating state — without requiring any domain-specific training data.

---

### 5.6 RAG Classifier

**Files:** `wave_soldering/rag_classifier.py` · `mv_conveyer/rag_classifier.py`

**The brain of the system.** Takes the 20+ features from one window and applies deterministic rules to produce RED, AMBER, or GREEN. No machine learning. No probability distributions. Just `if` statements with configurable thresholds.

**Wave Soldering Furnace — Decision Logic:**

```
IF rms_i_avg ≤ 2.0 A  AND  variance ≤ 1.0 A²
    → RED (Idle)
    Confidence based on: how far below 2.0 A the reading is

ELSE IF rms_i_avg ≥ 12.0 A
    AND (thd ≥ 0.15  OR  variance ≥ 5.0  OR  phase_imbalance ≥ 0.20)
    → GREEN (Load)
    Confidence based on: how many secondary conditions are met

ELSE IF rms_i_avg ≥ 14.4 A  (i.e., green_min × 1.2 — very high current)
    → GREEN (Load)
    Even without secondary conditions — current this high is unambiguous

ELSE
    → AMBER (No Load)
    The "everything else" bucket
```

**MV Conveyer — Decision Logic:**

```
IF i_avg < 0.10 A → RED (HALT)
IF i_avg ≥ 0.35 A AND (thd ≥ 0.05 OR variance ≥ 0.002 OR imbalance ≥ 0.15) → GREEN (RUNNING)
ELSE → AMBER (IDLE)
```

**Output:** A `ClassificationResult` containing `state`, `confidence` (0–1), `reason` (human-readable explanation), and `scores` (component scores for debugging).

> **Why secondary conditions for GREEN?** A furnace drawing 12 A could be in a transition between No Load and Load. By requiring *both* high current *and* at least one indicator of active processing (high THD, high variance, or phase imbalance), we reduce false GREEN classifications during ramp-up/ramp-down transitions.

> **Why the 1.2× override?** At 14.4 A, the furnace is unambiguously under load regardless of THD or variance. This prevents edge cases where a perfectly stable high-current draw (low variance, low THD) would incorrectly fall into AMBER.

---

### 5.7 State Manager

**Files:** `wave_soldering/state_manager.py` · `mv_conveyer/state_manager.py`

**The problem:** Without smoothing, classification flickers. If I_avg oscillates between 11.9 A and 12.1 A (right at the GREEN boundary), the raw output alternates AMBER → GREEN → AMBER every 30 seconds. This is technically correct but operationally meaningless.

**The solution — hysteresis with consecutive-window voting:**

A state change is only accepted after **N consecutive windows** produce the same new state (default N = 3). Until that threshold is reached, the system holds the previous state.

```
Window 1:  GREEN    streak = 1  → still AMBER (previous)
Window 2:  GREEN    streak = 2  → still AMBER
Window 3:  GREEN    streak = 3  → ACCEPTED → now GREEN
Window 4:  AMBER    streak = 1  → still GREEN
Window 5:  GREEN    streak = 1  → still GREEN (reset — direction changed)
Window 6:  AMBER    streak = 1  → still GREEN
Window 7:  AMBER    streak = 2  → still GREEN
Window 8:  AMBER    streak = 3  → ACCEPTED → now AMBER
```

**Physical significance:** At 60-second windows with 50% overlap, N = 3 means a state must persist for **at least 1.5 minutes** before being reported. This matches the physical reality — a furnace cannot meaningfully change from idle to production in under a minute. The hysteresis acts as a physical plausibility filter.

> **This is identical to how industrial PLC systems debounce digital inputs.** A noisy limit switch that bounces for 200 ms is filtered by requiring the signal to be stable for a configurable hold-off time. The same principle, applied to current-based state classification.

---

## 6. The ML Layer — iTransformer

### 6.1 Why Add ML When Rules Work?

The rule-based RAG classifier answers: "What is the machine doing *right now*?" But it cannot answer:

- **"Where are we in the current production cycle?"** (e.g., 70% through a heating cycle)
- **"What kind of job is running?"** (e.g., heavy vs. light PCB load)
- **"What will happen in the next 2 minutes?"** (predictive capability)

The iTransformer model is a multi-task deep-learning layer that adds these capabilities. It is *optional* — the rule-based system works independently. The ML model is trained on labels *generated from the same rules*, so it starts with the rules' knowledge and extends it with temporal patterns that rules cannot capture.

### 6.2 Auto-Label Generation

**Files:** `wave_soldering/ml/label_generator.py` · `mv_conveyer/ml/label_generator.py`

**No manual annotation.** Labels are generated automatically through a 6-step pipeline:

1. **Threshold-based state labelling:** Apply the same RMS thresholds as the RAG classifier to every sample → RED / AMBER / GREEN per sample
2. **Label smoothing:** Run-length encode the labels and merge any run shorter than 10 samples into its neighbours. Iterate until stable. This removes single-sample noise bursts.
3. **Cycle detection:** Find complete production cycles — contiguous non-RED segments that contain at least one GREEN sample, bounded by RED (idle) runs.
4. **Cycle position assignment:** Within each detected cycle, assign a linear position from 0.0 (start) to 1.0 (end). Samples outside any cycle get position = -1 (masked during training).
5. **Cycle feature extraction:** For each cycle, compute 6 features: log(duration), mean/max/std of I_avg during GREEN, green fraction, amber fraction.
6. **Job type clustering:** Run K-Means (default K=3) on cycle features. Sort cluster centres by ascending mean I_avg for stable labelling across runs. Each cycle gets a job-type label (e.g., "Light Load", "Medium Load", "Heavy Load").

> **Why auto-labelling?** In industrial settings, manually labelling 179,000 samples is impractical. By deriving labels from physics-based thresholds and unsupervised clustering, the system bootstraps ML training data from the same rules that drive the RAG classifier. The ML model then learns *temporal patterns* (how states evolve over time) that flat thresholds cannot capture.

### 6.3 iTransformer Architecture

**Files:** `wave_soldering/ml/itransformer.py` · `mv_conveyer/ml/itransformer.py`

The **iTransformer (Inverted Transformer)** treats each current channel as a separate token — rather than treating each time step as a token (as in standard Transformers). This is a key design choice for multivariate time series.

**Why inverted?**

In a standard Transformer applied to 4-channel time series of length 120, you would have 120 tokens (one per time step), each of dimension 4. Self-attention would model relationships *between time steps*. In the iTransformer, you have **4 tokens** (one per channel — I1, I2, I3, I_avg), each of dimension 120 (the full time series). Self-attention models relationships *between channels*.

**Physical motivation:** The relationship between I1, I2, and I3 tells you about load balance and phase health. The relationship between individual phases and I_avg tells you about anomalies. These cross-channel relationships are more informative for state classification than raw temporal attention over 120 time steps.

**Architecture:**

```
Input: (batch, 120, 4)  ← 120 seconds × 4 channels
           │
           ▼
   VariateEmbedding:  4 tokens, each (120,) → (d_model,)
   [Linear(120→64) + LayerNorm + Dropout]
           │
           ▼
   3 × iTransformerEncoderLayer:
   [Multi-Head Self-Attention (4 heads) over 4 variate tokens]
   [Feed-Forward(64→128→64)]
   [Residual + LayerNorm]
           │
           ▼
   Flatten: (4 × 64) = 256 → Linear(256 → 128) → ReLU
           │
           ├──► State Head:  Linear(128→64→3)    → softmax → RED/AMBER/GREEN
           ├──► Cycle Head:  Linear(128→32→1)    → sigmoid → [0, 1]
           └──► Job Head:    Linear(128→64→K)    → softmax → job type
```

**Default hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `seq_len` | 120 | 2 minutes of context at 1 Hz — enough to capture a full furnace heater cycle |
| `n_vars` | 4 | I1, I2, I3, I_avg |
| `d_model` | 64 | Compact embedding — the signal is low-dimensional |
| `n_heads` | 4 | One head per variate (I1, I2, I3, I_avg) |
| `n_layers` | 3 | Sufficient depth for cross-channel reasoning |
| `d_ff` | 128 | 2× model dim — standard Transformer ratio |
| `dropout` | 0.1 | Mild regularisation |

### 6.4 Multi-Task Training

**Files:** `wave_soldering/ml/trainer.py` · `mv_conveyer/ml/trainer.py`

The model is trained on three tasks simultaneously:

```
L_total = 1.0 × CrossEntropy(state) + 0.5 × MSE(cycle_position) + 0.5 × CrossEntropy(job_type)
```

- **State task:** Predict RED / AMBER / GREEN at the current time step
- **Cycle position task:** Predict how far through the current production cycle we are (0.0 → 1.0). Samples outside a cycle are masked (loss ignored).
- **Job type task:** Predict what kind of job is running (cluster label from K-Means). Also masked outside cycles.

**Training details:**

| Setting | Value |
|---|---|
| Optimiser | Adam (lr = 1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | 7 epochs without validation loss improvement |
| Gradient clipping | max_norm = 1.0 |
| Data split | 70% train / 15% validation / 15% test (temporal, no shuffle) |
| Normalisation | Z-score per channel, computed on training data only |
| Epochs | 25 (typical early-stop at ~15) |

**Checkpoints** are saved to `{machine}/ml/checkpoints/model.pt` and include the model weights, hyperparameters, normalisation statistics, label metadata, and training history.

### 6.5 Inference

**Files:** `wave_soldering/ml/predictor.py` · `mv_conveyer/ml/predictor.py`

The `FurnacePredictor` class wraps the trained model for batch inference:

1. Load checkpoint (model + normalisation stats)
2. Create sliding windows of `look_back` (120) samples using numpy stride tricks (memory-efficient, no copying)
3. Normalise using saved training statistics (prevents data leakage)
4. Run forward pass in batches of 256
5. Map outputs to human-readable labels

**Output columns added to the DataFrame:**

| Column | Type | Description |
|---|---|---|
| `ml_state` | int | 0 = RED, 1 = AMBER, 2 = GREEN |
| `ml_state_name` | str | "RED" / "AMBER" / "GREEN" |
| `ml_state_conf` | float | Softmax confidence (0–1) |
| `ml_cycle_pos` | float | Position in current cycle (0.0–1.0) |
| `ml_job_type` | int | Cluster index (0, 1, 2, ...) |
| `ml_job_name` | str | "Light Load" / "Medium Load" / "Heavy Load" / etc. |

> The first `look_back - 1` rows (119 samples) have NaN predictions — the model needs at least 2 minutes of history to make its first prediction.

---

## 7. The UI — Three Pages

### Page 1 — Data Validation

*"Is this data good enough to trust?"*

1. **Upload** — CSV file uploader (single file for furnace, two files for conveyer)
2. **Overview metrics** — sample count, duration, sample rate, I_avg range
3. **Time range selector** — slider to analyse a subset (e.g., first 4 hours)
4. **Raw time-series plots** — Phase currents I1/I2/I3 and I_avg over time
5. **Grid frequency chart** — 50 Hz reference line with ±0.5 Hz band highlighted
6. **Run validation** — processes all windows through the 4-check validation pipeline
7. **Validation metrics** — total/valid windows, valid %, average frequency, average imbalance
8. **FFT inspector** — select any window and see its frequency spectrum with dominant frequency highlighted
9. **Final verdict** — ✅ VALID DATA or ❌ INVALID DATA
10. **Proceed button** — only active if data passes (prevents bad data from reaching RAG)

### Page 2 — RAG State Visualisation

*"What was the machine doing, and when?"*

**For Wave Soldering Furnace (window-based):**
- Playback controls (Play/Pause/Reset/Speed/Jump)
- Three state circles (RED/AMBER/GREEN) — active one glows
- Live feature panel (RMS, THD, imbalance, variance)
- Window status (VALID/INVALID, timestamps, sample count)
- Zoomed time-series with current window highlighted in yellow
- FFT spectrum for current window
- Full-dataset state timeline with playback cursor
- Summary counts per state

**For MV Conveyer (per-sample, no windowing):**

The conveyer's ~15 s cycle is too fast for window-based classification — any window longer than ~3 s averages across ON and OFF states, making everything look GREEN. Instead, Page 2 classifies each 1-second sample directly from its I_avg value. Playback scrubs through samples at 1 Hz with:
- State circles + live I_avg display
- Per-sample state counts
- ±60 s cycle-detail view (phase currents, I_avg coloured by state, binary state indicator)
- Full-dataset overview with downsampled I_avg and state indicator

### Page 3 — ML Diagnostics

*"Can we learn deeper patterns?"*

Four tabs:

1. **Data & Labels** — auto-label generation UI, distribution histograms, cycle statistics, job cluster visualisation
2. **Train Model** — hyperparameter inputs, train button, live loss curves, early stopping display
3. **Evaluation** — confusion matrices, classification reports, cycle position MAE, per-class accuracy
4. **Inference View** — ML predictions overlaid on raw signal, ML vs. rule-based comparison, cycle position & job type timelines

---

## 8. Machine-Specific Differences

| Aspect | Wave Soldering Furnace | MV Conveyer |
|---|---|---|
| **Timestamp format** | MM:SS.f (wrapping, custom parser) | ISO datetime (standard) |
| **Data files** | Single CSV | Two CSVs (stop-and-run + cyclic) |
| **Current range** | 0.2 – 24 A | 0.10 – 0.79 A |
| **RED threshold** | I_avg ≤ 2.0 A | I_avg < 0.10 A |
| **GREEN threshold** | I_avg ≥ 12.0 A + secondary conditions | I_avg ≥ 0.35 A + secondary conditions |
| **Phase imbalance tolerance** | CV ≤ 0.80 | CV ≤ 1.50 (motor Phase 2 drops near zero during idle) |
| **Default window size** | 60 s | 3 s |
| **State labels** | Idle / No Load / Load | HALT / IDLE / RUNNING |
| **RAG Page 2 approach** | Window-based pipeline | Per-sample classification (cycle too fast for windows) |
| **ML label thresholds** | RED < 2.0 A, GREEN ≥ 12.0 A | RED < 0.10 A, GREEN ≥ 0.35 A |
| **iTransformer model** | Identical architecture | Identical architecture |

---

## 9. Every Threshold, Explained

All thresholds are configurable in the sidebar. Here is what each one means physically.

### Validation Thresholds

| Setting | Default (Furnace / Conveyer) | What it controls | Physical reasoning |
|---|---|---|---|
| Freq tolerance | ±0.5 Hz / ±0.5 Hz | How far from 50 Hz is acceptable | Grid frequency rarely deviates by more than ±0.2 Hz. 0.5 Hz gives generous margin for sensor jitter while still catching real faults. |
| Max phase imbalance | 0.80 CV / 1.50 CV | Maximum coefficient of variation across I1, I2, I3 | Healthy 3-phase loads balance within CV < 0.3. The furnace threshold (0.80) catches dead-phase faults. The conveyer threshold (1.50) is higher because motor I2 legitimately drops near zero during idle. |
| Valid window fraction | 80% / 80% | Minimum % of windows that must pass all checks | Allows up to 20% of windows to fail (sensor glitches, brief grid events) without rejecting the entire dataset. Below 80%, systematic data quality issues are likely. |

### RAG Classification Thresholds

| Setting | Default (Furnace / Conveyer) | Physical reasoning |
|---|---|---|
| RED max RMS | 2.0 A / 0.10 A | Below this, only standby electronics draw current. The machine is off or idle. |
| GREEN min RMS | 12.0 A / 0.35 A | Above this (with secondary conditions), the machine is actively processing product. |
| GREEN min THD | 0.15 / 0.05 | Active production creates complex current patterns (heater cycling, motor load changes) that increase harmonic distortion. |
| GREEN min variance | 5.0 A² / 0.002 A² | Under load, current fluctuates as the process demands change second by second. |
| GREEN min imbalance | 0.20 CV / 0.15 CV | Load naturally creates some phase imbalance as different heater zones or motor phases engage unevenly. |

---

## 10. Frequently Asked Questions

**"Why can't the FFT see 50 Hz?"**

Because the sensor gives us one reading per second (1 Hz sampling). The Nyquist theorem states you need *at least* 2× the frequency to detect it — so 100 readings per second for 50 Hz. At 1 Hz, the maximum detectable frequency is 0.5 Hz. The 50 Hz grid frequency is validated separately using the `Frequency` column that the sensor itself reports.

**"Why no machine learning for the primary classification?"**

Three reasons: (1) ML requires labelled training data, which doesn't exist for most factory machines. (2) ML models drift over time and need retraining. (3) ML cannot explain *why* it classified a window as GREEN — a technician cannot audit a 64-dimensional weight matrix. The rule-based system traces every decision to a number and a threshold: "RMS = 14.2 A > 12.0 A threshold, therefore GREEN." The ML layer is offered as an *optional* addition for cycle tracking and job typing, but the primary classification remains rule-based.

**"Why sliding windows with overlap instead of non-overlapping blocks?"**

Without overlap, a state transition happening in the middle of a window gets averaged into a blended state that may not match either side. With 50% overlap and a 60 s window, the step size is 30 s — so transitions are detected within ±30 s. The trade-off: more windows to process (2×), but much better temporal resolution at boundaries.

**"Why does the conveyer use per-sample classification instead of windows?"**

The conveyer cycles every ~15 seconds (11 s ON + 4 s OFF). Even a 3-second window averages 3 samples — and if 2 of those 3 straddle the ON/OFF boundary, the average pulls toward GREEN, hiding the IDLE gap. Per-sample classification at 1 Hz preserves the ON/OFF structure perfectly.

**"What happens if I tune the thresholds wrong?"**

The system degrades gracefully. If you set RED too high, idle periods get classified as AMBER. If GREEN too low, No Load periods get classified as GREEN. The state timeline on Page 2 makes miscalibration immediately obvious — you'll see patterns that don't match the raw current trace. Tuning is intentionally exposed in the sidebar so you can adjust and see the effect in real time.

**"Why hysteresis with 3 windows and not 1 or 10?"**

Three windows at 60 s with 50% overlap means a state must persist for ~1.5 minutes. Furnace state changes (idle → warm-up → production) take minutes. One window (30 s) is too sensitive to noise; 10 windows (5 minutes) is too sluggish and would delay reporting a genuine state change. Three is the empirically tested sweet spot for this equipment.

**"How does the ML model avoid data leakage?"**

Three safeguards: (1) The temporal split preserves time order — the model never sees future data during training. (2) Z-score normalisation statistics are computed on *training data only* and frozen for validation/test. (3) Labels are generated from the same deterministic rules applied to the raw signal, not from any retrospective annotation.

**"Can I add a third machine?"**

Yes. Duplicate either `wave_soldering/` or `mv_conveyer/`, adjust the thresholds in `rag_classifier.py`, `validation.py`, and `ml/label_generator.py`, then register the new machine in `MACHINE_META` inside `app.py`. The architecture is designed for this extension.

---

## 11. How to Run

### Local Development

**1. Install dependencies** (only needed once):
```bash
pip install -r requirements.txt
```

**2. Start the app:**
```bash
streamlit run app.py
```

**3. Open your browser** at `http://localhost:8501`

**4. Upload data:**
- **Wave Soldering:** `data/Current Data Final - Soldering Furnace - Sheet1.csv`
- **MV Conveyer:** both `data/mv_conveyor_updated_data/Stop_and_Run_MV_Conveyer.csv` and `data/mv_conveyor_updated_data/Cyclic_Run_MV_Conveyer.csv`

> **Tip:** Use the time-range slider on Page 1 to analyse a subset (e.g., first 4 hours) for a faster experience. The full 50-hour furnace dataset is automatically sub-sampled (capped at 1000 windows) for performance.

### Deployment

The app is deployed on **Streamlit Community Cloud**, which automatically rebuilds and redeploys on every push to the `main` branch of the GitHub repository — similar to how Vercel works for web apps. No manual deployment step is needed.

---

## 12. File Reference

```
Current_Sensor/
│
├── app.py                          Streamlit UI — 3-page front-end (Validation, RAG, ML)
├── requirements.txt                Python dependencies
├── README.md                       This file
│
├── wave_soldering/                 Wave Soldering Furnace pipeline
│   ├── __init__.py
│   ├── data_loader.py              CSV parsing, wrapping-timestamp reconstruction, i_avg
│   ├── windowing.py                Time-based 60s sliding windows, 50% overlap
│   ├── fft_analysis.py             Hann-windowed FFT, harmonics, THD, spectral energy
│   ├── validation.py               4-check validation (freq, range, phase, energy)
│   ├── feature_engineering.py      20+ features: RMS, variance, THD, imbalance, spectral
│   ├── rag_classifier.py           Rule-based RED / AMBER / GREEN (Idle / No Load / Load)
│   ├── state_manager.py            Hysteresis anti-flicker (3-window minimum hold)
│   └── ml/
│       ├── __init__.py
│       ├── label_generator.py      Auto-label: thresholds → smoothing → cycles → K-Means
│       ├── dataset.py              PyTorch Dataset, z-score normalisation, temporal split
│       ├── itransformer.py         iTransformer: variate-token Transformer (4 channels)
│       ├── trainer.py              Multi-task training: state + cycle_pos + job_type
│       ├── predictor.py            Batch inference with stride-trick sliding windows
│       ├── ui.py                   Streamlit ML page (4 tabs)
│       └── checkpoints/            Saved model weights & metadata
│
├── mv_conveyer/                    MV Conveyer pipeline (same structure, different thresholds)
│   ├── __init__.py
│   ├── data_loader.py              ISO datetime parsing, multi-file concatenation
│   ├── windowing.py                3s windows (fits inside 4s IDLE gap)
│   ├── fft_analysis.py             Identical to wave_soldering
│   ├── validation.py               Relaxed phase imbalance (1.50 CV)
│   ├── feature_engineering.py      Identical to wave_soldering
│   ├── rag_classifier.py           Thresholds: RED < 0.10 A, GREEN ≥ 0.35 A
│   ├── state_manager.py            Identical to wave_soldering
│   └── ml/
│       ├── __init__.py
│       ├── label_generator.py      Thresholds: HALT < 0.10 A, RUN ≥ 0.35 A
│       ├── dataset.py              Identical to wave_soldering
│       ├── itransformer.py         Identical to wave_soldering
│       ├── trainer.py              Identical to wave_soldering
│       ├── predictor.py            Identical to wave_soldering
│       ├── ui.py                   Conveyer-specific labels and color scheme
│       └── checkpoints/            Saved model weights & metadata
│
└── data/
    ├── Current Data Final - Soldering Furnace - Sheet1.csv
    └── mv_conveyor_updated_data/
        ├── Stop_and_Run_MV_Conveyer.csv
        └── Cyclic_Run_MV_Conveyer.csv
```

---

**Live App → [rakshit9695-current-sensor-project-app-qinted.streamlit.app](https://rakshit9695-current-sensor-project-app-qinted.streamlit.app/)**