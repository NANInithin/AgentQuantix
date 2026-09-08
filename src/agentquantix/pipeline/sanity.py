"""Will llama.cpp actually load this GGUF, answered from its header alone.

This exists because of a run that produced thirty files nobody can open.

`Nex-N2.5-mini` is a `qwen35moe` with `block_count = 41` and
`nextn_predict_layers = 1`. The converter wrote 733 tensors covering blocks
0-39 and stopped: the NextN / multi-token-prediction block is not emitted. The
metadata still promises 41. llama.cpp's loader walks `0..block_count-1` and
aborts on the first tensor it cannot find:

    check_tensor_dims: tensor 'blk.40.attn_norm.weight' not found
    llama_model_load_from_file_impl: failed to load model

The trap is that `llama-quantize` does NOT hit this. It streams tensors and
rewrites them; it never builds an inference graph, so it happily produced a
BF16 and every quant below it. Only `llama-imatrix` failed, because it is the
first step that actually loads the model — and the pipeline treated that as
"imatrix unavailable, carry on with the types that do not need one".

So the sweep ran to completion and published a repo of unloadable files. The
BF16 is broken, and so is every quant cut from it, because they all inherit
the same metadata and the same missing block.

Checking this needs no inference and no memory: the tensor names and the
`block_count` are both in the header. It is a few milliseconds against a
multi-hour run, so it happens BEFORE the BF16 is uploaded rather than after.

Deliberately conservative. A check that cannot run returns None — "no reason
found" — because refusing to publish a model on the strength of a header we
failed to parse would be a worse failure than the one being prevented.
"""

from __future__ import annotations

import re

BLOCK = re.compile(r"blk\.(\d+)\.")


def _value(field):
    """One metadata value out of a gguf ReaderField, across gguf versions.

    The `contents()` accessor is recent; older releases expose only `parts`
    and `data`, and the two disagree often enough that reading a header should
    not depend on which one is installed.
    """
    if field is None:
        return None
    contents = getattr(field, "contents", None)
    if callable(contents):
        try:
            return contents()
        except Exception:
            pass
    try:
        part = field.parts[field.data[0]]
    except Exception:
        return None
    try:
        if part.dtype.kind in "iuf":
            return part[0].item()
        return bytes(part).decode("utf-8", errors="replace")
    except Exception:
        return None


def block_coverage(path):
    """(declared_block_count, present_block_indices) for a GGUF, or (None, set()).

    None means the question could not be answered — the gguf package is
    missing, the header is unreadable, or the architecture does not use the
    `blk.N.` naming. Never an exception: this informs a decision, it is not
    the decision.
    """
    try:
        import gguf
    except ImportError:
        return None, set()

    try:
        reader = gguf.GGUFReader(str(path))
        architecture = _value(reader.fields.get("general.architecture"))
        if not isinstance(architecture, str) or not architecture:
            return None, set()
        declared = _value(reader.fields.get(f"{architecture}.block_count"))
        present = {int(match.group(1))
                   for tensor in reader.tensors
                   if (match := BLOCK.match(tensor.name))}
    except Exception:
        return None, set()

    return (int(declared) if isinstance(declared, (int, float)) else None,
            present)


def nextn_layers(path):
    """How many trailing blocks are multi-token-prediction heads.

    These are counted in `block_count` and deliberately NOT exported by the
    converter, so their absence is normal and expected rather than a defect.
    """
    try:
        import gguf
        reader = gguf.GGUFReader(str(path))
        architecture = _value(reader.fields.get("general.architecture"))
        if not isinstance(architecture, str):
            return 0
        value = _value(reader.fields.get(f"{architecture}.nextn_predict_layers"))
        return int(value) if isinstance(value, (int, float)) else 0
    except Exception:
        return 0


def missing_blocks(path):
    """Blocks the header promises that carry no tensors, EXCLUDING NextN heads.

    The NextN exclusion is the whole reason this function is subtle, and
    leaving it out cost a false positive that blocked a perfectly good model.
    `Nex-N2.5-mini` declares 41 blocks, exports 40, and sets
    `nextn_predict_layers = 1`: block 40 is a multi-token-prediction head that
    the converter is right not to emit. Another publisher quantized the same
    checkpoint the same day, imatrix and IQ types included, so a llama.cpp
    build that loads it plainly exists.

    What remains reportable is a block missing for no declared reason — a hole
    in the middle, or more absent blocks than the NextN count explains. That
    is still worth catching, but it is NOT the same question as "will this
    load", and nothing here should be mistaken for an answer to that. Only
    loading the model answers that, which is why the pipeline now runs the
    imatrix before it publishes anything.
    """
    declared, present = block_coverage(path)
    if not declared or not present:
        return []
    expected_absent = set(range(declared - nextn_layers(path), declared))
    return [index for index in range(declared)
            if index not in present and index not in expected_absent]


def unloadable_reason(path):
    """Why llama.cpp will refuse to load this file, or None if it looks fine.

    Only reports what is CERTAIN to fail. A missing block is certain: the
    loader requires every block in `0..block_count-1` and aborts on the first
    one absent, which no runtime works around.
    """
    missing = missing_blocks(path)
    if not missing:
        return None

    declared, _ = block_coverage(path)
    shown = ", ".join(str(index) for index in missing[:5])
    if len(missing) > 5:
        shown += f", ... ({len(missing)} in total)"
    return (f"the header declares {declared} blocks but carries no tensors "
            f"for block(s) {shown}, and no nextn_predict_layers value "
            "accounts for them. llama.cpp walks every block in "
            f"0..{declared - 1} and aborts on the first one absent")
