"""Lossless LoRA key mapping; tensor payloads are never changed."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
SOURCE_PREFIX = "base_model.model.model.layers."
DEST_PREFIX = "base_model.model.model.language_model.layers."
KEY = re.compile(
    r"^(\d+)\.(linear_attn|self_attn)\."
    r"(in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj|q_proj|k_proj|v_proj|o_proj)"
    r"\.lora_([AB])\.weight$"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result

def header(path):
    with Path(path).open("rb") as f:
        prefix = f.read(8)
        require(len(prefix) == 8, "Truncated safetensors length")
        n = struct.unpack("<Q", prefix)[0]
        require(0 < n <= 10_000_000, "Unexpected safetensors header size")
        raw = f.read(n)
        require(len(raw) == n, "Truncated safetensors header")
        data = json.loads(raw, object_pairs_hook=unique_object)
    intervals = []
    for key, value in data.items():
        if key == "__metadata__":
            continue
        require(value["dtype"] == "F32", f"Expected source F32 tensor: {key}")
        shape = value["shape"]
        require(len(shape) == 2 and all(isinstance(x, int) and x > 0 for x in shape),
                f"Unexpected tensor shape: {key}")
        start, end = value["data_offsets"]
        require(start >= 0 and end - start == 4 * math.prod(shape), f"Invalid offsets: {key}")
        intervals.append((start, end))
    offset = 0
    for start, end in sorted(intervals):
        require(start == offset, "Gaps or overlaps in tensor payload")
        offset = end
    require(8 + n + offset == Path(path).stat().st_size, "Payload length mismatch")
    return data, 8 + n

def renamed_header(source, expected_count):
    result, pairs = {}, {}
    for key, spec in source.items():
        if key == "__metadata__":
            result[key] = spec
            continue
        require(key.startswith(SOURCE_PREFIX), f"Unexpected/already-mapped prefix: {key}")
        suffix = key[len(SOURCE_PREFIX):]
        match = KEY.fullmatch(suffix)
        require(match is not None, f"Unexpected target tensor: {key}")
        leaf = suffix.rsplit(".lora_", 1)[0]
        side = match.group(4)
        require(spec["shape"][0 if side == "A" else 1] == 32, f"Rank changed: {key}")
        pairs.setdefault(leaf, set()).add(side)
        mapped = DEST_PREFIX + suffix
        require(mapped not in result, "Key collision")
        result[mapped] = spec
    require(len(result) - int("__metadata__" in result) == expected_count, "Tensor count changed")
    require(all(sides == {"A", "B"} for sides in pairs.values()), "Missing A/B pair")
    return result

def equal_payloads(source, derived):
    a, ao = header(source)
    b, bo = header(derived)
    h = hashlib.sha256()
    with Path(source).open("rb") as left, Path(derived).open("rb") as right:
        left.seek(ao); right.seek(bo)
        while True:
            x, y = left.read(1024 * 1024), right.read(1024 * 1024)
            require(x == y, "Tensor bytes changed during mapping")
            if not x:
                break
            h.update(x)
    return h.hexdigest()

def convert(source_adapter, destination, *, expected_count=304):
    source_adapter, destination = Path(source_adapter), Path(destination)
    require(not destination.resolve().is_relative_to(source_adapter.resolve()),
            "Destination must not be inside original adapter")
    source_weights = source_adapter / "adapter_model.safetensors"
    original, offset = header(source_weights)
    mapped = renamed_header(original, expected_count)
    destination.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(mapped, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    target = destination / "adapter_model.safetensors"
    with source_weights.open("rb") as src, target.open("xb") as dst:
        dst.write(struct.pack("<Q", len(raw))); dst.write(raw)
        src.seek(offset)
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    config = source_adapter / "adapter_config.json"
    with config.open("rb") as src, (destination / config.name).open("xb") as dst:
        shutil.copyfileobj(src, dst)
    require(header(target)[0] == mapped, "Derived header failed round-trip")
    return {
        "tensor_count": expected_count,
        "tensor_payload_sha256": equal_payloads(source_weights, target),
        "source_adapter_sha256": sha(source_weights),
        "derived_adapter_sha256": sha(target),
        "adapter_config_sha256": sha(config),
        "source_key_prefix": SOURCE_PREFIX, "destination_key_prefix": DEST_PREFIX,
        "tensor_values_changed": False, "merged": False,
    }

def verify_derived(source_adapter, destination, manifest):
    source_adapter, destination = Path(source_adapter), Path(destination)
    src, dst = source_adapter / "adapter_model.safetensors", destination / "adapter_model.safetensors"
    require(sha(src) == manifest["source_adapter_sha256"], "Original adapter changed")
    require(sha(dst) == manifest["derived_adapter_sha256"], "Derived adapter changed")
    require(header(dst)[0] == renamed_header(header(src)[0], manifest["tensor_count"]),
            "Mapping is not exactly the approved key rewrite")
    require(equal_payloads(src, dst) == manifest["tensor_payload_sha256"], "Payload hash changed")
    require(sha(source_adapter / "adapter_config.json") ==
            sha(destination / "adapter_config.json") == manifest["adapter_config_sha256"],
            "Adapter configuration changed")
