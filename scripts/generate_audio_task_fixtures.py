import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "src" / "agentquantix" / "fixtures" / "voice"
SEPARATION = FIXTURES / "separation"
SAMPLE_RATE = 44100


def read_wav(path):
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        rate = audio.getframerate()
        if audio.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM: {path}")
        samples = np.frombuffer(audio.readframes(audio.getnframes()),
                                dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def write_wav(path, samples, rate):
    values = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.rint(values * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm.tobytes())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_separation(jfk, jfk_rate):
    speech = resample_poly(jfk, SAMPLE_RATE // math.gcd(jfk_rate, SAMPLE_RATE),
                           jfk_rate // math.gcd(jfk_rate, SAMPLE_RATE))
    speech *= 0.12 / max(float(np.sqrt(np.mean(np.square(speech)))), 1e-12)
    times = np.arange(speech.size, dtype=np.float64) / SAMPLE_RATE
    accompaniment = (0.09 * (
        np.sin(2 * np.pi * 220.0 * times)
        + np.sin(2 * np.pi * 329.63 * times)
        + np.sin(2 * np.pi * 440.0 * times)) / math.sqrt(3.0))
    mixture = speech + accompaniment
    gain = min(1.0, 0.95 / max(float(np.max(np.abs(mixture))), 1e-12))
    speech *= gain
    accompaniment *= gain
    mixture *= gain
    paths = {
        "speech.wav": speech,
        "accompaniment.wav": accompaniment,
        "mixture.wav": mixture,
    }
    for name, samples in paths.items():
        write_wav(SEPARATION / name, samples, SAMPLE_RATE)
    return paths


def make_vc_reference(source, output):
    ratio = 2 ** (4 / 12)
    shifted_rate = round(16000 * ratio)
    speed = 16000 / shifted_rate
    filters = (f"asetrate={shifted_rate},aresample=16000,"
               f"atempo={speed:.12f},atrim=duration=4")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-af", filters,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
    ], check=True)


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    SEPARATION.mkdir(parents=True, exist_ok=True)
    source = FIXTURES / "jfk.wav"
    jfk, rate = read_wav(source)
    if rate != 16000:
        raise ValueError(f"expected 16 kHz JFK fixture, got {rate}")
    make_separation(jfk, rate)
    vc_reference = FIXTURES / "vc-reference.wav"
    make_vc_reference(source, vc_reference)
    names = ["separation/" + name
             for name in ("mixture.wav", "speech.wav", "accompaniment.wav")]
    names.append("vc-reference.wav")
    manifest = {
        "source_audio": "jfk.wav",
        "source_license": "public-domain",
        "separation_sample_rate": SAMPLE_RATE,
        "separation": {
            "speech_source": "JFK public-domain speech",
            "accompaniment": "synthetic 220, 329.63 and 440 Hz chord",
        },
        "voice_conversion_reference": {
            "pitch_shift_semitones": 4,
            "duration_seconds": 4,
            "method": "ffmpeg asetrate/aresample/atempo",
        },
        "numpy": np.__version__,
        "files": {
            name: {
                "bytes": (FIXTURES / name).stat().st_size,
                "sha256": sha256(FIXTURES / name),
            }
            for name in names
        },
    }
    (SEPARATION / "manifest.json").write_text(
        json.dumps({**manifest, "files": {
            name: manifest["files"][name]
            for name in names if name.startswith("separation/")}},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (FIXTURES / "audio-task-fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
