# 📡 Signal Analyzer – Communication Systems Lab

## 🚀 Overview

This project is a **real-time Communication Systems Simulator & Analyzer** that visualizes and compares different modulation techniques.

It allows users to interact with signals, observe waveform behavior, and evaluate system performance using key metrics.

---

## 🎯 Features

* 📊 Real-time waveform visualization (Message, Carrier, Modulated, Demodulated)
* 📈 Performance metrics:

  * SNR (Signal-to-Noise Ratio)
  * FOM (Figure of Merit)
  * NMSE (Error Analysis)
* 🧠 Automatic ranking of modulation techniques
* ⚙️ Block-level control:

  * Message
  * Modulator
  * Channel
  * Demodulator
* 🚨 Fault detection with diagnosis messages
* 📧 Email alert system for system failures

---

## 📡 Modulation Techniques Implemented

* AM (Amplitude Modulation)
* FM (Frequency Modulation)
* DSB-SC
* SSB-SC
* PAM
* PWM
* PPM

---

## 💡 Unique Concept

This project introduces a custom **Figure of Merit (FOM)** that combines:

* SNR (Signal Quality)
* NMSE (Signal Accuracy)
* Theoretical Gain (Real-world performance factor)

---

## 📊 Key Insight

The system demonstrates real-world ranking:

FM > DSB-SC > SSB-SC > AM > PAM > PWM > PPM

---

## 🛠️ Tech Stack

* Python (FastAPI)
* NumPy, SciPy (Signal Processing)
* HTML, CSS, JavaScript
* Plotly (Visualization)

---

## ▶️ How to Run

### 1. Backend

```bash
cd backend
uvicorn main:app --reload
```

### 2. Frontend

Open:

```
frontend/index.html
```

---

Then open http://localhost:5500 in your browser.

## ⚠️ Note

Email credentials are removed for security reasons.

---


## 📌 Author

**Taaha Hameed**
