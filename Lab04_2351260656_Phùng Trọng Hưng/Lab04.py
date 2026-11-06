
from pathlib import Path
import re
import csv
import json
import time
import hashlib
import wave

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    sosfiltfilt,
    resample_poly
)
from math import gcd


# ============================================================
# 1. CAU HINH
# ============================================================

ROOT = Path(__file__).resolve().parent

SR = 16000
SEED = 42

FOLDERS = [
    "units",
    "audio_output",
    "figures",
    "hmm_demo",
    "reports"
]

for folder in FOLDERS:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)


WORDS = [
    "mời", "khách", "hàng", "số",
    "không", "một", "hai", "ba",
    "bốn", "năm", "sáu", "bảy",
    "tám", "chín", "đến", "quầy",
    "xin", "vui", "lòng", "cảm", "ơn"
]

DIGIT = dict(zip(
    "0123456789",
    [
        "không", "một", "hai", "ba", "bốn",
        "năm", "sáu", "bảy", "tám", "chín"
    ]
))


# ============================================================
# 2. CAC HAM CO BAN
# ============================================================

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def normalize_queue_text(text):
    """Chuan hoa van ban va doc tung chu so."""

    text = text.lower()
    text = text.replace(".", " ").replace(",", " ")

    tokens = re.findall(
        r"\d+|[^\W\d_]+",
        text,
        flags=re.UNICODE
    )

    output = []

    for token in tokens:
        if token.isdigit():
            output.extend(DIGIT[c] for c in token)
        else:
            output.append(token)

    return output


def save_wav(path, audio):
    """Luu WAV PCM 16-bit bang thu vien chuan Python."""

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio)

    peak = np.max(np.abs(audio)) if len(audio) else 0

    if peak > 0.98:
        audio = audio * (0.98 / peak)

    pcm = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def load_unit(path):
    """Doc WAV PCM 16-bit bang thu vien chuan Python."""

    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        width = f.getsampwidth()
        fs = f.getframerate()
        frames = f.readframes(f.getnframes())

    if width != 2:
        raise ValueError("File WAV phai la PCM 16-bit")

    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    x /= 32768.0

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    if fs != SR:
        g = gcd(fs, SR)
        x = resample_poly(x, SR // g, fs // g)

    return x.astype(np.float32)


# ============================================================
# 3. SINH TIN HIEU AM HOC GIA LAP
# ============================================================

def stable_seed(word, candidate):
    """Tao seed co tinh lap lai cho tung tu."""

    value = f"{word}_{candidate}".encode("utf-8")
    digest = hashlib.sha256(value).hexdigest()

    return int(digest[:8], 16)


def generate_unit(word, candidate=1):
    """
    Tao mot acoustic unit gia lap.

    Tin hieu gom:
    - Thanh phan huu thanh: tong cac harmonic.
    - Thanh phan vo thanh: nhieu ngau nhien.
    - Bien do thay doi theo thoi gian.
    - Pho duoc tao voi cac vung formant mo phong.

    Day khong phai bo may doc tieng Viet.
    """

    rng = np.random.default_rng(
        stable_seed(word, candidate) + SEED
    )

    # Thoi luong khac nhau giua cac tu va candidate
    duration = rng.uniform(0.20, 0.48)

    if candidate == 2:
        duration *= 1.15

    n = int(duration * SR)

    t = np.arange(n) / SR

    # Tan so co ban
    base_f0 = rng.uniform(100, 210)

    if candidate == 2:
        base_f0 *= 1.12

    # Bien do bao
    attack = np.minimum(t / 0.025, 1.0)
    release = np.minimum(
        (duration - t) / 0.04,
        1.0
    )

    envelope = np.maximum(
        0,
        attack * release
    )

    envelope *= (
        0.75 + 0.25 * np.sin(
            2 * np.pi * 2.5 * t
        )
    )

    # Thanh phan huu thanh
    phase = 2 * np.pi * np.cumsum(
        np.full(n, base_f0) / SR
    )

    voiced = np.zeros(n)

    for harmonic in range(1, 9):
        amplitude = 1.0 / harmonic

        voiced += amplitude * np.sin(
            harmonic * phase
        )

    voiced /= 3.0

    # Thanh phan vo thanh
    noise = rng.normal(0, 0.12, n)

    # Loc nhieu de tao mau pho giong am thanh
    sos = butter(
        2,
        [300, 5000],
        btype="bandpass",
        fs=SR,
        output="sos"
    )

    noise = sosfiltfilt(sos, noise)

    # Tao ti le huu thanh / vo thanh theo tung tu
    ratio = rng.uniform(0.65, 0.95)

    signal = (
        ratio * voiced
        + (1 - ratio) * noise
    )

    signal *= envelope

    # Bien doi nhe de tao hai candidate khac nhau
    if candidate == 2:
        signal = np.roll(signal, int(0.002 * SR))

    signal = signal.astype(np.float32)

    signal *= 0.075 / max(rms(signal), 1e-8)

    return np.clip(
        signal,
        -0.98,
        0.98
    )


# ============================================================
# 4. XAY DUNG INVENTORY
# ============================================================

def make_inventory():

    rows = []
    inventory = {}

    for word in WORDS:

        inventory[word] = []

        for candidate in [1, 2]:

            path = (
                ROOT
                / "units"
                / f"{word}_{candidate:02d}.wav"
            )

            audio = generate_unit(
                word,
                candidate
            )

            save_wav(path, audio)

            inventory[word].append(path)

            rows.append({
                "label": word,
                "speaker": "Synthetic acoustic proxy",
                "duration_s": len(audio) / SR,
                "rms": rms(audio),
                "file_path": str(path.relative_to(ROOT)),
                "candidate": candidate,
                "sample_rate": SR,
                "data_type": "Artificial acoustic signal"
            })

    with open(
        ROOT / "metadata.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    return inventory, rows


# ============================================================
# 5. TARGET COST VA JOIN COST
# ============================================================

def join_cost(a, b, ms=25, nfft=512):

    n = max(
        32,
        int(SR * ms / 1000)
    )

    def feature(x):

        z = np.zeros(n, dtype=np.float32)

        q = x[-n:] if len(x) >= n else x

        z[-len(q):] = q

        spectrum = np.abs(
            np.fft.rfft(
                z * np.hanning(n),
                n=nfft
            )
        )

        return 20 * np.log10(
            np.maximum(spectrum, 1e-7)
        )

    return float(
        np.mean(
            (feature(a) - feature(b)) ** 2
        )
    )


def target_cost(x):

    duration = len(x) / SR

    return (
        ((duration - 0.32) / 0.25) ** 2
        +
        ((rms(x) - 0.075) / 0.08) ** 2
    )


# ============================================================
# 6. CROSSFADE VA NOI AM
# ============================================================

def crossfade(a, b, n):

    n = min(
        int(n),
        len(a),
        len(b)
    )

    if n <= 0:
        return np.concatenate([a, b])

    alpha = np.linspace(
        0,
        1,
        n,
        endpoint=True,
        dtype=np.float32
    )

    middle = (
        (1 - alpha) * a[-n:]
        +
        alpha * b[:n]
    )

    return np.concatenate([
        a[:-n],
        middle,
        b[n:]
    ])


def concatenate(xs, ms):

    if not xs:
        return np.array([], dtype=np.float32)

    y = xs[0]

    for x in xs[1:]:

        y = crossfade(
            y,
            x,
            int(SR * ms / 1000)
        )

    return y


# ============================================================
# 7. THU NGHIEM DURATION VA PITCH
# ============================================================

def change_duration(x, rate):
    """
    Thay doi thoi luong bang noi suy.
    rate > 1: nhanh hon, ngan hon.
    rate < 1: cham hon, dai hon.
    """

    new_length = max(
        1,
        int(len(x) / rate)
    )

    old_positions = np.arange(len(x))

    new_positions = np.linspace(
        0,
        len(x) - 1,
        new_length
    )

    return np.interp(
        new_positions,
        old_positions,
        x
    ).astype(np.float32)


def change_pitch(x, semitones):
    """
    Doi pitch bang resampling va dua ve thoi luong ban dau.

    Day la phep thu nghiem don gian, co the tao artifact.
    """

    factor = 2 ** (semitones / 12)

    new_length = max(
        1,
        int(len(x) / factor)
    )

    temp = np.interp(
        np.linspace(0, len(x) - 1, new_length),
        np.arange(len(x)),
        x
    )

    result = np.interp(
        np.linspace(0, len(temp) - 1, len(x)),
        np.arange(len(temp)),
        temp
    )

    return result.astype(np.float32)


# ============================================================
# 8. VE WAVEFORM VA SPECTROGRAM
# ============================================================

def plot_wave_spec(
    y,
    title,
    path,
    bounds=None
):

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(12, 5)
    )

    t = np.arange(len(y)) / SR

    ax[0].plot(
        t,
        y,
        linewidth=0.45
    )

    ax[0].set_title(
        title + " - Waveform"
    )

    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")

    if bounds:

        for boundary in bounds:
            ax[0].axvline(
                boundary,
                color="red",
                linestyle="--",
                alpha=0.6
            )

    ax[1].specgram(
        y,
        NFFT=512,
        Fs=SR,
        noverlap=384
    )

    ax[1].set_title("Spectrogram")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# 9. HMM-STYLE DEMO
# ============================================================

def hmm_demo(best, tokens):

    nframes = max(
        100,
        int(len(best) / SR * 100)
    )

    t = np.arange(nframes) * 0.01

    state = np.minimum(
        np.arange(nframes) * len(tokens) // nframes,
        len(tokens) - 1
    )

    f0 = np.where(
        state % 4 != 0,
        145 + 12 * np.sin(2 * np.pi * t / 1.8),
        0.0
    )

    energy = (
        0.35
        +
        0.12 * np.sin(2 * np.pi * t / 2.2)
    )

    formant = (
        650
        +
        100 * np.sin(2 * np.pi * t / 1.4)
    )

    np.savez(
        ROOT / "hmm_demo" / "hmm_tracks.npz",
        time=t,
        f0=f0,
        energy=energy,
        formant1=formant,
        state=state
    )

    # Tao tin hieu minh hoa tu cac tham so
    sample_count = int(nframes * SR / 100)

    tt = np.arange(sample_count) / SR

    f0i = np.interp(
        tt,
        t,
        np.maximum(f0, 100)
    )

    phase = 2 * np.pi * np.cumsum(f0i) / SR

    excitation = (
        np.sin(phase)
        +
        0.45 * np.sin(2 * phase)
    )

    excitation *= np.interp(
        tt,
        t,
        energy
    )

    sos = butter(
        2,
        [500, 1500],
        btype="bandpass",
        fs=SR,
        output="sos"
    )

    y = sosfiltfilt(
        sos,
        excitation
    )

    y = (
        0.8 * y
        /
        (np.max(np.abs(y)) + 1e-9)
    )

    return y.astype(np.float32), t, f0, energy, formant


# ============================================================
# 10. MAIN
# ============================================================

def main():

    start = time.perf_counter()

    # A. INVENTORY
    print("\n[A] Tao inventory...")

    inventory, metadata = make_inventory()

    sentence = "Mời khách hàng số 23 đến quầy 5."

    tokens = normalize_queue_text(sentence)

    print("Sentence:", sentence)
    print("Tokens:", tokens)

    missing = [
        word
        for word in tokens
        if word not in inventory
    ]

    if missing:
        raise ValueError(
            f"Inventory thieu token: {missing}"
        )

    candidates = [
        [
            load_unit(path)
            for path in inventory[word]
        ]
        for word in tokens
    ]

    # B. CONCATENATION
    print("\n[B] Noi cac unit...")

    baseline = [
        item[0]
        for item in candidates
    ]

    raw = np.concatenate(baseline)

    save_wav(
        ROOT / "audio_output" / "raw_concat.wav",
        raw
    )

    # C. UNIT SELECTION - DYNAMIC PROGRAMMING
    print("\n[C] Unit selection bang Dynamic Programming...")

    dp = np.full(
        (len(tokens), 2),
        np.inf
    )

    back = np.full(
        (len(tokens), 2),
        -1,
        dtype=int
    )

    for j, x in enumerate(candidates[0]):
        dp[0, j] = target_cost(x)

    lam = 0.35

    for i in range(1, len(tokens)):

        for j, x in enumerate(candidates[i]):

            for k, previous in enumerate(candidates[i - 1]):

                value = (
                    dp[i - 1, k]
                    +
                    target_cost(x)
                    +
                    lam * join_cost(previous, x)
                )

                if value < dp[i, j]:

                    dp[i, j] = value
                    back[i, j] = k

    j = int(np.argmin(dp[-1]))

    choice = [j]

    for i in range(len(tokens) - 1, 0, -1):

        choice.append(
            int(back[i, choice[-1]])
        )

    choice = choice[::-1]

    selected = [
        candidates[i][j]
        for i, j in enumerate(choice)
    ]

    best = concatenate(
        selected,
        15
    )

    save_wav(
        ROOT / "audio_output" / "best_concat.wav",
        best
    )

    # Thu nghiem crossfade
    print("\n[D] Crossfade...")

    for ms in [0, 5, 15, 30]:

        audio = concatenate(
            selected,
            ms
        )

        save_wav(
            ROOT / "audio_output" / f"concat_xfade_{ms}ms.wav",
            audio
        )

    # Duration
    print("\n[E] Duration va pitch...")

    for rate in [0.8, 1.0, 1.2]:

        audio = change_duration(
            best,
            rate
        )

        save_wav(
            ROOT / "audio_output" / f"duration_{rate:.1f}x.wav",
            audio
        )

    # Pitch
    for semitone in [-2, 0, 2]:

        audio = change_pitch(
            best,
            semitone
        )

        save_wav(
            ROOT / "audio_output" / f"pitch_{semitone:+d}st.wav",
            audio
        )

    # F. HMM DEMO
    print("\n[F] HMM-style parameter demo...")

    y_hmm, t, f0, energy, formant = hmm_demo(
        best,
        tokens
    )

    save_wav(
        ROOT / "audio_output" / "hmm_tts.wav",
        y_hmm
    )

    # ========================================================
    # 11. TAO CAC BIEU DO
    # ========================================================

    print("\n[G] Tao bieu do...")

    fig_dir = ROOT / "figures"

    # 1. Duration
    plt.figure(figsize=(9, 3))

    plt.hist(
        [r["duration_s"] for r in metadata],
        bins=12
    )

    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Inventory duration")

    plt.tight_layout()

    plt.savefig(
        fig_dir / "inventory_duration.png",
        dpi=150
    )

    plt.close()

    # 2. RMS
    plt.figure(figsize=(9, 3))

    plt.hist(
        [r["rms"] for r in metadata],
        bins=12
    )

    plt.xlabel("RMS")
    plt.ylabel("Count")
    plt.title("Inventory RMS")

    plt.tight_layout()

    plt.savefig(
        fig_dir / "inventory_rms.png",
        dpi=150
    )

    plt.close()

    # Vi tri ranh gioi
    def get_bounds(xs, ms):

        position = 0
        result = []

        n = int(SR * ms / 1000)

        for x in xs[:-1]:

            position += len(x) - n

            result.append(
                position / SR
            )

        return result

    # 3. Raw waveform + spectrogram
    plot_wave_spec(
        raw,
        "Raw concatenation",
        fig_dir / "raw_concat_wave_spec.png",
        get_bounds(baseline, 0)
    )

    # 4. Best waveform + spectrogram
    plot_wave_spec(
        best,
        "Unit selection + 15 ms crossfade",
        fig_dir / "best_concat_wave_spec.png",
        get_bounds(selected, 15)
    )

    # 5. Join cost
    join_rows = []

    for i in range(len(tokens) - 1):

        join_rows.append({
            "boundary": (
                f"{tokens[i]} -> {tokens[i + 1]}"
            ),
            "candidate_1_to_1": join_cost(
                candidates[i][0],
                candidates[i + 1][0]
            ),
            "selected": join_cost(
                selected[i],
                selected[i + 1]
            )
        })

    with open(
        ROOT / "reports" / "join_cost.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=join_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(join_rows)

    plt.figure(figsize=(11, 4))

    xx = np.arange(len(join_rows))

    plt.plot(
        xx,
        [r["candidate_1_to_1"] for r in join_rows],
        marker="o",
        label="Fixed candidate"
    )

    plt.plot(
        xx,
        [r["selected"] for r in join_rows],
        marker="s",
        label="DP selected"
    )

    plt.xticks(
        xx,
        [r["boundary"] for r in join_rows],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Join cost")
    plt.title("Join cost comparison")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fig_dir / "join_cost.png",
        dpi=150
    )

    plt.close()

    # 6. HMM tracks
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 5),
        sharex=True
    )

    ax[0].plot(t, f0)
    ax[0].set_ylabel("F0 (Hz)")
    ax[0].set_title("HMM-style parameter trajectories")

    ax[1].plot(
        t,
        formant,
        label="F1-like"
    )

    ax[1].plot(
        t,
        energy * 1000,
        label="Energy x1000"
    )

    ax[1].legend()
    ax[1].set_xlabel("Time (s)")

    plt.tight_layout()

    plt.savefig(
        fig_dir / "hmm_tracks.png",
        dpi=150
    )

    plt.close()

    # 7. Output comparison
    plt.figure(figsize=(8, 3))

    plt.bar(
        ["raw", "best", "HMM demo"],
        [
            len(raw) / SR,
            len(best) / SR,
            len(y_hmm) / SR
        ]
    )

    plt.ylabel("Duration (s)")
    plt.title("Output duration comparison")

    plt.tight_layout()

    plt.savefig(
        fig_dir / "output_comparison.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 12. BAO CAO KET QUA
    # ========================================================

    summary = {
        "sentence": sentence,
        "tokens": tokens,
        "inventory_units": len(metadata),
        "candidates_per_unit": 2,
        "sample_rate": SR,
        "selected_candidates_1based": [
            x + 1 for x in choice
        ],
        "raw_duration_s": len(raw) / SR,
        "best_duration_s": len(best) / SR,
        "hmm_duration_s": len(y_hmm) / SR,
        "crossfade_ms": [0, 5, 15, 30],
        "duration_rates": [0.8, 1.0, 1.2],
        "pitch_semitones": [-2, 0, 2],
        "runtime_s": time.perf_counter() - start
    }

    summary_path = (
        ROOT / "reports" / "run_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("HOAN THANH LAB 4")
    print("=" * 60)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\nCac thu muc ket qua:")

    for folder in FOLDERS:
        print(" -", ROOT / folder)

if __name__ == "__main__":
    main()
