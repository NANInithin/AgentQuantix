"""Generate tiny, original CC0 geometric/PCM fixtures without model weights."""
import json
import math
from pathlib import Path
import struct
import wave
import zlib

ROOT = Path(__file__).resolve().parents[1] / "src/agentquantix/fixtures/media"


def png(path, phase):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    pixels = bytearray()
    for y in range(64):
        pixels.append(0)
        for x in range(64):
            color = (225, 220, 205)
            if (x - (20 + phase))**2 + (y - 24)**2 < 120:
                color = (210, 55, 45)
            if 35 <= x <= 53 and 35 <= y <= 53:
                color = (45, 85, 210)
            pixels.extend(color)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    png(ROOT / "reference.png", 0)
    png(ROOT / "first.png", 0)
    png(ROOT / "last.png", 12)
    with wave.open(str(ROOT / "reference.wav"), "wb") as audio:
        audio.setparams((2, 2, 32000, 32000, "NONE", "not compressed"))
        audio.writeframes(b"".join(struct.pack("<hh", int(8000 * math.sin(2 * math.pi * 440 * n / 32000)),
                                               int(7000 * math.sin(2 * math.pi * 550 * n / 32000))) for n in range(32000)))
    prompts = [{"id": "shapes", "text": "A red circle to the left of a blue square on a cream background. No people."},
               {"id": "counting", "text": "Three yellow cubes on a plain blue table. No people."},
               {"id": "landscape", "text": "A small stone bridge over a clear stream, green trees, soft daylight. No people."},
               {"id": "multilingual", "text": "Un cercle rouge et deux carrés bleus. 赤い円と青い四角。"}]
    (ROOT / "fixtures.json").write_text(json.dumps({"license": "CC0-1.0", "origin": "original procedural AgentQuantix fixtures",
        "prompts": prompts, "images": ["reference.png", "first.png", "last.png"], "audio": "reference.wav",
        "image_size": [64, 64], "audio_sample_rate": 32000, "audio_channels": 2,
        "audio_duration_seconds": 1}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
