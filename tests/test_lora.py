#!/usr/bin/env python3
"""End-to-end LoRA fix exerciser. Run against a live tt-metal server.

  SDXL_LORA=<path> WAN_LORA=<path> TT_SERVER_URL=http://127.0.0.1:8000 \
      venv/bin/python test_lora.py

Needs `requests`, so use ComfyUI's venv rather than the system python.

SDXL_LORA must point at an adapter that carries text-encoder tensors -- most
published SDXL LoRAs are UNet-only, and against those the clip-scale assertions
cannot pass. The Bug 1 test detects this and skips rather than reporting a
spurious failure. Check an adapter with:

  python3 -c "import json,struct,sys; f=open(sys.argv[1],'rb'); \
h=json.loads(f.read(struct.unpack('<Q',f.read(8))[0])); \
print(any('text_encoder' in k or k.startswith('lora_te') for k in h))" <lora>

Bug 1 (SDXL): verified from the response body's `lora` status object.
Bug 2 (WAN): no lora status in response — HTTP 200 + server log check required.
  Watch: grep -iE "Activated LoRA|Clearing active LoRA" <server-log>
"""
import os
import sys
import json
import requests

URL = os.environ.get("TT_SERVER_URL", "http://127.0.0.1:8000")
SDXL = os.environ.get("SDXL_LORA")
WAN = os.environ.get("WAN_LORA")
TIMEOUT = 1800


def post(path, body):
    r = requests.post(f"{URL}{path}", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def lora_has_text_encoder(path):
    """True if the adapter actually carries text-encoder tensors.

    Most published SDXL LoRAs are UNet-only. Against one of those the
    text_encoder assertions below can never pass -- there is nothing to fuse --
    so the split-scale test needs an adapter that has them.
    """
    import json
    import struct

    try:
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    except Exception as e:
        print(f"  WARN: could not read {path}: {e}")
        return None
    return any(k.startswith("lora_te") or "text_encoder" in k for k in header if k != "__metadata__")


def sdxl_denoise(scale_unet=None, scale_clip=None, scale=None):
    body = {
        "prompt": "a pixel art castle",
        "num_inference_steps": 4,
        "guidance_scale": 5.0,
        "seed": 0,
        "lora_path": SDXL,
    }
    if scale_unet is not None:
        body["lora_scale_unet"] = scale_unet
    if scale_clip is not None:
        body["lora_scale_clip"] = scale_clip
    if scale is not None:
        body["lora_scale"] = scale
    return post("/latent/denoise", body).get("lora") or {}


def test_bug1():
    print("Bug 1 (SDXL split scales):")
    if not SDXL:
        print("  SKIP: set SDXL_LORA env var")
        return True
    if lora_has_text_encoder(SDXL) is False:
        print(f"  SKIP: {SDXL} is UNet-only (no text-encoder tensors).")
        print("        The clip-scale assertions need an adapter that has them,")
        print("        otherwise text_encoder=False is correct, not a regression.")
        return True
    ok = True

    s = sdxl_denoise(scale_unet=0.0, scale_clip=1.0)
    print(f"   unet=0.0, clip=1.0 → {json.dumps(s)}")
    ok &= check("unet=False", s.get("unet") is False)
    ok &= check("text_encoder=True", s.get("text_encoder") is True)

    s = sdxl_denoise(scale_unet=1.0, scale_clip=0.0)
    print(f"   unet=1.0, clip=0.0 → {json.dumps(s)}")
    ok &= check("unet=True", s.get("unet") is True)
    ok &= check("text_encoder=False", s.get("text_encoder") is False)

    s = sdxl_denoise(scale=0.5)
    print(f"   legacy scale=0.5  → {json.dumps(s)}")
    ok &= check("both applied", s.get("unet") and s.get("text_encoder"))
    ok &= check("scale_unet=0.5", s.get("scale_unet") == 0.5)
    ok &= check("scale_clip=0.5", s.get("scale_clip") == 0.5)

    return ok


def wan_denoise(scale):
    body = {
        "prompt": "a cinematic city at night",
        "num_inference_steps": 2,
        "high_lora_path": WAN,
        "lora_scale": scale,
        "seed": 0,
    }
    r = requests.post(f"{URL}/video/denoise", json=body, timeout=TIMEOUT)
    return r.status_code


def test_bug2():
    print("Bug 2 (WAN lora_scale=0 disables):")
    if not WAN:
        print("  SKIP: set WAN_LORA env var")
        return True
    ok = True

    code = wan_denoise(1.0)
    ok &= check(f"scale=1.0 returns 200 (got {code})", code == 200)

    code = wan_denoise(0.0)
    ok &= check(f"scale=0.0 returns 200 (got {code})", code == 200)
    print("  NOTE: confirm server log shows 'Clearing active LoRA (lora_scale=0.0)'")
    print("        and NO 'Activated LoRA' for the scale=0.0 call.")

    return ok


if __name__ == "__main__":
    results = [test_bug1(), test_bug2()]
    print()
    print("RESULT:", "ALL PASS" if all(results) else "FAILURES PRESENT")
    sys.exit(0 if all(results) else 1)
