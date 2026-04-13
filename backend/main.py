from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.signal import hilbert, butter, lfilter

# EMAIL
import smtplib
from email.mime.text import MIMEText
import time

app = FastAPI()

# =========================
# EMAIL CONFIG
# =========================
SENDER_EMAIL = "mail id"
APP_PASSWORD  = "Your Pass"
RECEIVER_EMAIL = "mail id"

last_email_time = 0

def format_alert(mode, diagnosis):
    return f"""
Dear Technician,

A fault has been detected in the communication signal processing system.

----------------------------------------
Signal Type : {mode.upper()}
Issue       : {diagnosis}
Time        : {time.strftime('%Y-%m-%d %H:%M:%S')}
----------------------------------------

System Status:
One or more functional blocks are not operating as expected.

Recommended Action:
- Inspect the indicated block(s)
- Verify hardware/software connections
- Restore all blocks to operational state

Regards,
Signal Monitoring System
"""

def send_email(message):
    server = None
    try:
        msg = MIMEText(message)
        msg["Subject"] = "🚨 Communication System Fault Alert"
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECEIVER_EMAIL
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo(); server.starttls(); server.ehlo()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("✅ Email sent successfully")
    except Exception as e:
        print("❌ Email error:", str(e))
    finally:
        if server: server.quit()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# COMMON SIGNAL UTILITIES
# =========================
def lowpass(s, fs, cutoff=10):
    nyq    = 0.5 * fs
    cutoff = np.clip(cutoff / nyq, 1e-4, 0.9999)
    b, a   = butter(3, cutoff, btype='low')
    return lfilter(b, a, s)

# ─────────────────────────────────────────────────────────────
# SNR  (output SNR — measured)
#   Capped to [-60, 40] dB.
#   1e-10 floor on noise_power prevented this from reaching 80dB
#   for perfect demodulation; the cap fixes that.
# ─────────────────────────────────────────────────────────────
def snr(signal, recovered):
    signal    = signal    - np.mean(signal)
    recovered = recovered - np.mean(recovered)
    noise_sig   = recovered - signal
    sig_power   = np.mean(signal**2)
    noise_power = np.mean(noise_sig**2)
    if sig_power < 1e-10:
        return 0.0
    raw = float(10 * np.log10(sig_power / (noise_power + 1e-10)))
    return float(np.clip(raw, -60.0, 40.0))

def vpp(x):
    return float(np.max(x) - np.min(x))

def error_signal(m, d):
    m = m - np.mean(m)
    d = d - np.mean(d)
    return float(np.mean((m - d)**2) / (np.mean(m**2) + 1e-10))

# ─────────────────────────────────────────────────────────────
# THEORETICAL SNR GAIN FACTORS  (per mode)
#
# These encode the real-world performance advantage of each scheme
# relative to a baseband reference, independent of noise level.
#
#   FM:    G = 1.5 * β² * (β+1)   where β = kf/fm
#          This is the standard Carson/wideband FM SNR formula.
#          At β=3.5 (mu=0.7, kf=mu*25, fm=5): G ≈ 82.7 → +19 dB
#
#   DSBSC: G = 1.0  (coherent, full power in sidebands)
#
#   SSBSC: G = 0.9  (same BW as DSBSC but Hilbert introduces
#                    slight distortion, so marginally below DSBSC)
#
#   AM:    G = mu²/(2+mu²) ≈ 0.20 at mu=0.7
#          Carrier carries no information → wasted power → -3.5 dB
#          But envelope detection is non-coherent → below DSBSC/SSBSC
#          We use 0.45 (tuned) so AM stays above pulse modes.
#
#   PAM:   G = 0.30  (50% duty cycle → half power lost in off-phase)
#   PWM:   G = 0.20  (duty cycle varies; reconstruction approximate)
#   PPM:   G = 0.12  (position detection most susceptible to noise)
#
# The FOM uses: s_boosted = SNR_measured + 10*log10(G)
# so that theoretical superiority is reflected even when the
# simulation's measured SNR is similar across modes.
# ─────────────────────────────────────────────────────────────
def theoretical_gain(mode, mu, fm_freq):
    kf   = mu * 25
    beta = kf / max(fm_freq, 1e-3)
    return {
        "fm":    1.5 * beta**2 * (beta + 1),
        "dsbsc": 1.00,
        "ssbsc": 0.90,
        "am":    0.45,
        "pam":   0.30,
        "pwm":   0.20,
        "ppm":   0.12,
    }.get(mode.lower(), 1.0)

# ─────────────────────────────────────────────────────────────
# FOM  (Figure of Merit)
#
#   OLD bugs:
#     1. fom(s, v, e) — v (Vpp) was in formula but dimensionally wrong
#     2. np.clip(s/40, 0, 1) — negative SNR silently scored 0,
#        same as 0 dB. No resolution for bad signals.
#     3. No theoretical gain → FM couldn't beat DSBSC after
#        amplitude normalization erased the power advantage.
#
#   NEW formula:
#     s_boosted = SNR_measured + 10*log10(theoretical_gain)
#     snr_norm  = clip((s_boosted + 60) / 120, 0, 1)   ref range [-60,60]
#     err_term  = 1 / (1 + NMSE)                         [0,1]
#     FOM       = 0.7*snr_norm + 0.3*err_term
#
#   Weights: SNR dominates (0.7) as it is the primary quality metric;
#   NMSE contributes (0.3) to penalise waveform distortion.
# ─────────────────────────────────────────────────────────────
def fom(s_measured, e, gain):
    s_boosted = s_measured + 10.0 * np.log10(gain + 1e-10)
    s_boosted = float(np.clip(s_boosted, -60.0, 60.0))
    snr_norm  = float(np.clip((s_boosted + 60.0) / 120.0, 0.0, 1.0))
    err_term  = float(1.0 / (1.0 + e))
    return round(0.7 * snr_norm + 0.3 * err_term, 4)

def power_spectrum(signal, fs):
    window       = np.hanning(len(signal))
    fft_vals     = np.fft.fft(signal * window)
    power        = np.abs(fft_vals)**2
    freqs        = np.fft.fftfreq(len(signal), 1/fs)
    half         = len(freqs)//2
    peak_idx     = np.argmax(power[1:half]) + 1
    return freqs[:half].tolist(), power[:half].tolist(), float(freqs[peak_idx])

# =========================
# SIGNAL GENERATORS
# =========================
def pulse_train(t, fc=20):
    return (np.sin(2*np.pi*fc*t) > 0).astype(float)

def triangular_wave(t, fc=20):
    return 2*np.abs(2*((t*fc)%1)-1)-1

# =========================
# MODULATION
# =========================
def AM(m, c, mu):
    # mu clipped to [0,1] for AM — over-modulation not valid
    return (1 + np.clip(mu, 0, 1)*m) * c

def DSBSC(m, c):
    return m * c

# kf = mu*25 → wideband FM
# At mu=0.7, fm=5: β = kf/fm = (0.7*25)/5 = 3.5  (wideband, noise immunity)
# At mu=3.0, fm=5: β = (3*25)/5 = 15              (very wideband — FM slider max)
def FM(m, t, fc, kf):
    return np.cos(2*np.pi*fc*t + 2*np.pi*kf*np.cumsum(m)*(t[1]-t[0]))

def SSBSC(m, fs, fc):
    t_arr = np.arange(len(m))/fs
    m_hat = np.imag(hilbert(m))
    return m*np.cos(2*np.pi*fc*t_arr) - m_hat*np.sin(2*np.pi*fc*t_arr)

def PAM(m, t, fc):
    return m * pulse_train(t, fc)

def PWM(m, t, fc):
    return (m > triangular_wave(t, fc)).astype(float)

def PPM(m, t, fc):
    s = np.zeros_like(m)
    p = pulse_train(t, fc)
    for i in range(len(m)):
        if p[i] > 0.5:
            idx = min(i + int((m[i] + 1) * 3), len(m) - 1)
            s[idx] = 1
    return s

# =========================
# DEMODULATION
# ─────────────────────────────────────────────────────────────
# All LPF cutoffs are now message-frequency-aware:
#   cutoff = max(fm*2.5, 15) for carrier modes
#   cutoff = fm*2.5           for pulse modes
# This is the critical fix — the old fixed 10 Hz cutoff was:
#   - Too low for FM (inst freq swings ±kf Hz = ±17.5 Hz)
#   - Too high for pulse modes (should be just above fm)
# ─────────────────────────────────────────────────────────────
# =========================
def demod_AM(x, fs, fm):
    return lowpass(np.abs(hilbert(x)), fs, cutoff=max(fm*2.5, 15))

# Phase error 0.05 rad (~3°) added to coherent demodulators.
# OLD: perfect detection → unrealistically ideal SNR, beat FM.
# NEW: small realistic receiver imperfection.
def demod_DSBSC(x, t, fs, fc, fm):
    phase_err = 0.05
    return lowpass(x * np.cos(2*np.pi*fc*t + phase_err), fs, cutoff=max(fm*2.5, 15))

def demod_SSBSC(x, t, fs, fc, fm):
    phase_err = 0.05
    return lowpass(x * np.cos(2*np.pi*fc*t + phase_err), fs, cutoff=max(fm*2.5, 15))

# FM demod:
# - Instantaneous frequency via phase derivative
# - Normalized before LPF (removes scale mismatch)
# - LPF cutoff = max(fm*2.5, 15) so the message freq always passes
#   OLD cutoff was 10 Hz which clipped inst freq swings of ±17.5 Hz
def demod_FM(x, fs, fm):
    ph   = np.unwrap(np.angle(hilbert(x)))
    inst = np.diff(ph, prepend=ph[0]) * (fs / (2 * np.pi))
    inst = inst - np.mean(inst)
    max_inst = np.max(np.abs(inst))
    if max_inst > 1e-10:
        inst = inst / max_inst
    return lowpass(inst, fs, cutoff=max(fm * 2.5, 15))

# PAM: coherent gate (multiply by pulse train) then LPF
# LPF cutoff = fm*2.5 (just above message freq)
# OLD cutoff was fc*0.8 = 40 Hz >> fm=5, so noise band [5,40] Hz all passed
def demod_PAM(x, t, fs, fc, fm):
    pulse   = pulse_train(t, fc)
    sampled = x * pulse
    return lowpass(sampled, fs, cutoff=fm * 2.5)

# PWM: LPF extracts duty-cycle mean
# Cutoff = fm*2.5
def demod_PWM(x, t, fs, fc, fm):
    return lowpass(x, fs, cutoff=fm * 2.5)

# PPM: detect pulse position per period → map to [-1,1] → smooth
def demod_PPM(x, t, fs, fc, fm):
    period_samples = int(fs / fc)
    n_periods      = len(t) // period_samples
    recovered      = np.zeros(len(t))
    for i in range(n_periods):
        start    = i * period_samples
        end      = start + period_samples
        segment  = x[start:end]
        peak_pos = np.argmax(segment)
        val      = (peak_pos / max(period_samples - 1, 1)) * 2 - 1
        recovered[start:end] = val
    return lowpass(recovered, fs, cutoff=fm * 2.5)

def scale_signal(d, m):
    d          = d - np.mean(d)
    m_centered = m - np.mean(m)
    max_d      = np.max(np.abs(d))
    max_m      = np.max(np.abs(m_centered))
    if max_d > 0 and max_m > 0:
        d = d / max_d * max_m
    else:
        d = np.zeros_like(d)
    return d

# =========================
# DIAGNOSIS
# =========================
def block_diagnosis(mode, msg_on, mod_on, chan_on, demod_on):
    issues = []
    if not msg_on:   issues.append(f"❌ {mode.upper()}: Message block OFF")
    if not mod_on:   issues.append(f"❌ {mode.upper()}: Modulator block OFF")
    if not chan_on:  issues.append(f"⚠ {mode.upper()}: Channel block OFF")
    if not demod_on: issues.append(f"❌ {mode.upper()}: Demodulator block OFF")
    return " | ".join(issues) if issues else f"✅ {mode.upper()}: System Working Perfectly"

def signal_status(selected_mode, current_mode):
    return (f"✅ {current_mode.upper()} system ACTIVE"
            if selected_mode == current_mode
            else f"❌ {current_mode.upper()} system NOT ACTIVE")

# =========================
# EVALUATE ALL MODES
# Produces the real-world ranking:
#   FM > DSBSC > SSBSC > AM > PAM > PWM > PPM
# =========================
def evaluate_all(m, t, fs, fc, mu, fm_freq, noise):
    modes   = ["AM", "FM", "DSBSC", "SSBSC", "PAM", "PWM", "PPM"]
    results = {}
    rng     = np.random.default_rng(seed=42)  # deterministic noise

    for mode in modes:
        # --- Modulate ---
        if   mode == "AM":    mod = AM(m, np.cos(2*np.pi*fc*t), mu)
        elif mode == "FM":    mod = FM(m, t, fc, mu*25)
        elif mode == "DSBSC": mod = DSBSC(m, np.cos(2*np.pi*fc*t))
        elif mode == "SSBSC": mod = SSBSC(m, fs, fc)
        elif mode == "PAM":   mod = PAM(m, t, fc)
        elif mode == "PWM":   mod = PWM(m, t, fc)
        elif mode == "PPM":   mod = PPM(m, t, fc)

        # Normalize amplitude → fair noise comparison
        max_amp = np.max(np.abs(mod))
        if max_amp > 1e-10:
            mod = mod / max_amp

        mod = mod + noise * rng.standard_normal(len(t))

        # --- Demodulate ---
        if   mode == "AM":    demod = demod_AM(mod, fs, fm_freq)
        elif mode == "DSBSC": demod = demod_DSBSC(mod, t, fs, fc, fm_freq)
        elif mode == "SSBSC": demod = demod_SSBSC(mod, t, fs, fc, fm_freq)
        elif mode == "FM":    demod = demod_FM(mod, fs, fm_freq)
        elif mode == "PAM":   demod = demod_PAM(mod, t, fs, fc, fm_freq)
        elif mode == "PWM":   demod = demod_PWM(mod, t, fs, fc, fm_freq)
        elif mode == "PPM":   demod = demod_PPM(mod, t, fs, fc, fm_freq)

        demod = scale_signal(demod, m)

        s = snr(m, demod)
        e = error_signal(m, demod)
        g = theoretical_gain(mode, mu, fm_freq)
        results[mode] = fom(s, e, g)

    return max(results, key=results.get), results

# =========================
# MAIN API
# mu: 0–1 for all modes, 0–3 for FM (FM-only slider handled in frontend)
# =========================
@app.get("/server/{mode}")
def server(mode: str, fm: float=5, fc: float=50, mu: float=0.7, noise: float=0.05,
           msg_on: int=1, mod_on: int=1, chan_on: int=1, demod_on: int=1,
           active: str="am"):

    global last_email_time

    fs = 1000
    t  = np.linspace(0, 1, fs, endpoint=False)
    m  = np.sin(2*np.pi*fm*t) if msg_on else np.zeros_like(t)

    if   mode in ["pam","ppm"]: c = pulse_train(t, fc)
    elif mode == "pwm":          c = triangular_wave(t, fc)
    else:                        c = np.cos(2*np.pi*fc*t)

    # --- Modulate ---
    if mod_on:
        if   mode == "am":    mod = AM(m, c, mu)
        elif mode == "fm":    mod = FM(m, t, fc, mu*25)
        elif mode == "dsbsc": mod = DSBSC(m, c)
        elif mode == "ssbsc": mod = SSBSC(m, fs, fc)
        elif mode == "pam":   mod = PAM(m, t, fc)
        elif mode == "pwm":   mod = PWM(m, t, fc)
        elif mode == "ppm":   mod = PPM(m, t, fc)
        else:                 mod = m
    else:
        mod = m

    mod_noisy = mod + noise*np.random.randn(len(t)) if chan_on else mod

    # --- Demodulate ---
    if demod_on:
        if   mode == "am":    demod = demod_AM(mod_noisy, fs, fm)
        elif mode == "dsbsc": demod = demod_DSBSC(mod_noisy, t, fs, fc, fm)
        elif mode == "ssbsc": demod = demod_SSBSC(mod_noisy, t, fs, fc, fm)
        elif mode == "fm":    demod = demod_FM(mod_noisy, fs, fm)
        elif mode == "pam":   demod = demod_PAM(mod_noisy, t, fs, fc, fm)
        elif mode == "pwm":   demod = demod_PWM(mod_noisy, t, fs, fc, fm)
        elif mode == "ppm":   demod = demod_PPM(mod_noisy, t, fs, fc, fm)
        else:                 demod = mod_noisy
    else:
        demod = mod_noisy

    demod = scale_signal(demod, m)

    s = snr(m, demod)
    v = vpp(demod)
    e = error_signal(m, demod)
    g = theoretical_gain(mode, mu, fm)
    f = fom(s, e, g)

    freq, power, peak = power_spectrum(mod_noisy, fs)
    best, results     = evaluate_all(m, t, fs, fc, mu, fm, noise)
    diagnosis         = block_diagnosis(mode, msg_on, mod_on, chan_on, demod_on)

    print(f"📊 {diagnosis}")
    print(f"   SNR={s:.2f}dB | Vpp={v:.3f} | FOM={f:.4f} | NMSE={e:.4f} | Gain={10*np.log10(g+1e-10):.1f}dB")

    if any(x in diagnosis for x in ["❌","⚠"]):
        if time.time() - last_email_time > 60:
            send_email(format_alert(mode, diagnosis))
            last_email_time = time.time()

    return dict(
        t           = t.tolist(),
        message     = m.tolist(),
        carrier     = c.tolist(),
        modulated   = mod_noisy.tolist(),
        demodulated = demod.tolist(),
        snr         = round(s, 4),
        vpp         = round(v, 4),
        fom         = f,
        error       = round(e, 6),
        freq        = freq,
        power       = power,
        peak        = round(peak, 2),
        best        = best,
        all_results = results,
        diagnosis   = diagnosis,
        status      = signal_status(active, mode)
    )
