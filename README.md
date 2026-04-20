# Soldering Furnace Current Analyser

A Streamlit web application that reads electrical current data from a soldering furnace, validates the signal quality, and automatically classifies the furnace's operating state in real time — with no machine learning involved. Every decision the system makes is transparent, explainable, and based on configurable rules.

---

## What does this application actually do?

A soldering furnace draws electrical current from a 3-phase AC power supply. How much current it draws — and the pattern in which it draws it — tells you exactly what the furnace is doing at any given moment:

| State | What it means | Current signature |
|---|---|---|
| 🔴 **RED — Idle** | Furnace is off or in standby | Very low current across all phases |
| 🟡 **AMBER — No Load** | Furnace is powered, heaters cycling, but no product inside | Moderate current, regular cycling pattern |
| 🟢 **GREEN — Load** | Furnace running with product — full production | High current, sustained draw |

This application reads a CSV file of current measurements, checks that the data is trustworthy, then plays through it window by window — showing you which state the furnace was in at every point in time.

---

## The Data

The CSV file contains readings from a 3-phase current sensor installed on the furnace. The sensor captures one reading every second (1 Hz) and records:

| Column | What it is |
|---|---|
| `TimeStamp` | Time of measurement (MM:SS format, wraps every 60 minutes) |
| `I1` | RMS current on Phase 1 (Amperes) |
| `I2` | RMS current on Phase 2 (Amperes) |
| `I3` | RMS current on Phase 3 (Amperes) |
| `Frequency` | Measured AC grid frequency (should be ~50 Hz) |

> **Important:** These are **RMS (Root Mean Square)** values — not raw waveform samples. The sensor has already done the heavy lifting of converting the raw AC signal into a meaningful average current value. This shapes every technical decision in the app.

The dataset covers approximately **50 hours** of furnace operation (~179,000 readings), including active production periods and a shutdown at the end.

---

## How the Application is Structured

The project is split into 8 focused Python modules. Each has one job and one job only. Here is how data flows through them:

```
CSV File
   │
   ▼
data_loader.py       ← "Clean the raw data"
   │
   ▼
windowing.py         ← "Slice into time chunks"
   │
   ├──► fft_analysis.py      ← "Analyse the frequency pattern"
   │
   ├──► validation.py        ← "Is this window trustworthy?"
   │
   ├──► feature_engineering.py  ← "Compute meaningful numbers"
   │
   ├──► rag_classifier.py    ← "Decide: RED, AMBER, or GREEN?"
   │
   └──► state_manager.py     ← "Smooth out rapid flickering"
   
   │
   ▼
app.py               ← "Show everything in the browser"
```

---

## Module-by-Module Walkthrough

### `data_loader.py` — Clean the raw data

**The problem it solves:** The timestamp in the CSV (`32:32.1`) is in MM:SS format and wraps back to `00:00.0` every 60 minutes. Over 50 hours, this means dozens of resets. The loader detects each wrap and adds an hour offset, reconstructing a clean, monotonically increasing timeline.

It also:
- Renames columns to consistent lowercase names (`I1` → `i1`)
- Computes `i_avg` (the average of all three phases) since it is not in the source CSV
- Ensures all current values are positive numbers

---

### `windowing.py` — Slice into time chunks

**Why this exists:** Looking at one data point tells you almost nothing. Looking at 60 data points together tells you whether the furnace is running hard, idling, or cycling. This module groups consecutive readings into overlapping **windows**.

- **Default window size:** 60 seconds (60 samples)
- **Overlap:** 50% — so consecutive windows share half their data, giving smooth transitions
- **Time-based:** Windows are defined by actual timestamps, not row numbers, so they work correctly even if the sensor skips a beat

---

### `fft_analysis.py` — Analyse the frequency pattern

**What FFT means in plain English:** FFT (Fast Fourier Transform) answers the question: *"Is there a repeating pattern in this data, and how fast does it repeat?"*

**The important adaptation for this dataset:** Because the sensor gives us one reading per second (1 Hz), we cannot detect the 50 Hz power line directly — that would need at least 100 readings per second. Instead, the FFT here analyses the *operational cycling* of the furnace: the heaters switch on and off roughly every 4–5 seconds, which shows up as a peak at ~0.25 Hz (once every 4 seconds). This is exactly the kind of pattern that distinguishes "furnace running" from "furnace idle".

The 50 Hz grid frequency is validated separately using the `Frequency` column that the sensor reports directly.

**Features extracted per window:**
- Dominant cycling frequency (the main repeating pattern)
- Energy at the fundamental frequency and its harmonics
- Total spectral energy
- High-frequency energy ratio

---

### `validation.py` — Is this window trustworthy?

Before classifying a window, the system checks whether the data in it is physically meaningful. A window is marked **VALID** only if all four checks pass:

1. **Grid frequency check** — The `Frequency` column should read ~50 Hz (±0.5 Hz). If the grid is unstable or the sensor has lost lock, this fails.
2. **Current range check** — All current values must be positive and finite. At least one phase must be carrying real current.
3. **Phase consistency check** — The three phases should carry roughly similar currents. An extreme imbalance (e.g. one phase completely dead while others are high) suggests a sensor fault or wiring issue.
4. **Energy floor check** — The average current must be above a minimum threshold to distinguish real signal from sensor noise.

At the dataset level, if fewer than 80% of windows pass, the entire dataset is flagged as **INVALID** and the RAG analysis is blocked.

---

### `feature_engineering.py` — Compute meaningful numbers

This module turns a window of 60 current readings into a compact set of numbers that describe what the furnace was doing:

**Time-domain features** (computed directly from the current values):
- **RMS** per phase and average — how hard the furnace is working overall
- **Variance** — how much the current fluctuates within the window
- **Peak-to-peak** — the range from lowest to highest reading
- **Phase imbalance** — how unevenly load is distributed across the three phases

**Frequency-domain features** (from the FFT):
- Energy at the fundamental cycling frequency
- Energy at the 2nd and 3rd harmonic
- Total spectral energy
- High-frequency energy ratio

**Derived features:**
- **THD (Total Harmonic Distortion proxy)** — ratio of harmonic energy to fundamental energy; a proxy for how distorted or irregular the current waveform is
- **High-frequency ratio** — fraction of energy in the upper frequency band

---

### `rag_classifier.py` — Decide: RED, AMBER, or GREEN?

This is the brain of the system. It takes the features from one window and applies a strict set of human-readable rules to assign a state. **No machine learning. No black boxes.**

```
IF average RMS < 2.0 A  AND  variance is low
    → RED (Idle)

ELSE IF average RMS > 12.0 A  AND  (THD is high OR variance is high OR phase imbalance is high)
    → GREEN (Load)

ELSE
    → AMBER (No Load)
```

Every threshold is configurable in the sidebar — you can tune them to match your specific furnace without touching any code.

---

### `state_manager.py` — Smooth out rapid flickering

**The problem:** The raw classification can flicker rapidly between states when the furnace is near a threshold boundary. If the current sits at exactly 12.1 A, it might alternate GREEN → AMBER → GREEN every window, which is noisy and meaningless.

**The solution — hysteresis:** A new state is only officially adopted after it appears in **3 consecutive windows**. This means brief excursions across a boundary are ignored, and only genuine sustained state changes are reported. This is the same principle used in industrial PLC control systems.

---

### `app.py` — Show everything in the browser

The Streamlit front-end, structured into two pages accessible from the sidebar.

---

## The Two Pages

### Page 1 — Data Validation

This page answers: *"Is this data good enough to trust?"*

You upload the CSV here. The page immediately shows you:

1. **Raw time-series plots** — see all three phases and the average current plotted over time
2. **Grid frequency chart** — see whether the measured frequency stays near 50 Hz throughout
3. **FFT inspector** — pick any window and inspect its operational frequency spectrum; see the dominant cycling frequency highlighted
4. **Validation metrics** — percentage of valid windows, average grid frequency, average phase imbalance
5. **Final verdict** — a clear ✅ VALID DATA or ❌ INVALID DATA badge

The **"Proceed to RAG Analysis"** button only becomes active if the data passes validation. You cannot accidentally analyse bad data.

---

### Page 2 — RAG State Visualisation

This page answers: *"What was the furnace doing, and when?"*

It simulates time-moving playback through the dataset, window by window.

**Controls:**
- ▶ Play / ⏸ Pause — step through windows automatically
- Speed slider — control how fast playback moves
- Jump-to-window slider — scrub to any point manually

**What you see at each window:**

| Panel | What it shows |
|---|---|
| **State circles** | Three coloured circles (RED / AMBER / GREEN). The active one glows brightly. The others are dimmed. |
| **Feature panel** | Live values for RMS, THD, phase imbalance, and variance for the current window |
| **Window status** | VALID or INVALID badge, window start/end time, sample count |
| **Live time-series** | Zoomed view centred on the current window, with the window highlighted in yellow |
| **FFT spectrum** | Operational frequency spectrum for the current window |
| **State timeline** | Horizontal coloured bar showing RED / AMBER / GREEN across the full dataset, with a cursor marking the current position |
| **Summary counts** | Total windows in each state across the full dataset |

---

## How to Run

**1. Install dependencies** (only needed once):
```bash
pip install -r requirements.txt
```

**2. Start the app:**
```bash
cd C:\Users\DELL\Desktop\Current_Sensor
python -m streamlit run app.py
```

**3. Open your browser** at `http://localhost:8501`

**4. Upload the CSV** on Page 1:
```
data/Current Data Final - Soldering Furnace - Sheet1.csv
```

> **Tip:** Use the time-range slider on Page 1 to analyse a subset (e.g. first 4 hours) for a faster experience. The full 50-hour dataset is processed by sub-sampling windows automatically.

---

## Configurable Thresholds (Sidebar)

Everything is tunable without touching code:

| Setting | What it controls | Default |
|---|---|---|
| Window size | How many seconds per analysis chunk | 60 s |
| Freq tolerance | How far from 50 Hz is still "OK" | ±0.5 Hz |
| Max phase imbalance | CV threshold for phase consistency check | 0.80 |
| Valid window fraction | % of windows that must pass for VALID dataset | 80% |
| RED max RMS | Upper RMS boundary for Idle state | 2.0 A |
| GREEN min RMS | Lower RMS boundary for Load state | 12.0 A |
| GREEN min THD | THD level that contributes to GREEN | 0.15 |
| GREEN min imbalance | Imbalance level that contributes to GREEN | 0.20 |

---

## Key Technical Choices Explained Simply

**"Why not 50 Hz FFT on the current data?"**
The sensor gives one reading per second. To see 50 Hz, you'd need at least 100 readings per second. Instead, the FFT reveals the furnace's *operational rhythm* — how often the heaters switch on and off — which is just as useful for classification.

**"Why no machine learning?"**
Machine learning requires labelled training data, is a black box, and cannot be easily audited. This system uses explicit rules: every decision traces back to a number and a threshold. A technician can read the logic and understand exactly why a window was classified GREEN.

**"Why sliding windows with overlap?"**
A single data point is noisy. Averaging over 60 seconds smooths out noise. The 50% overlap means state transitions are captured at half the window resolution (30 seconds) rather than missing events that fall at a window boundary.

---

## File Reference

```
Current_Sensor/
├── app.py                    Streamlit UI — 2-page front-end
├── data_loader.py            CSV parsing, timestamp reconstruction, i_avg computation
├── windowing.py              Time-based sliding window generator
├── fft_analysis.py           FFT engine and spectral feature extraction
├── validation.py             Per-window and dataset-level quality checks
├── feature_engineering.py   RMS, variance, THD, imbalance, spectral features
├── rag_classifier.py         Rule-based RED / AMBER / GREEN classifier
├── state_manager.py          Hysteresis anti-flicker state machine
├── requirements.txt          Python package dependencies
└── data/
    └── Current Data Final - Soldering Furnace - Sheet1.csv
```
