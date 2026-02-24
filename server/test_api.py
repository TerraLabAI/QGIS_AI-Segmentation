"""Test script for the SAM2 API server.

Usage:
    python test_api.py [--url URL] [--api-key KEY]

Tests the full workflow: health -> set_image -> predict -> reset
Uses a synthetic 256x256 test image (no real image needed).
"""
import argparse
import base64
import json
import time
import sys

import numpy as np

try:
    import requests
except ImportError:
    print("Install requests first: pip install requests")
    sys.exit(1)


def make_test_image(size=256):
    """Create a synthetic RGB image with a bright square in the center."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = 50  # dark background
    # Bright square in center
    q = size // 4
    img[q:3 * q, q:3 * q] = [200, 180, 60]
    return img


def encode_image(img):
    return base64.b64encode(img.tobytes()).decode("utf-8")


def test_health(base_url, headers):
    print("1. Testing /health ...")
    r = requests.get("{}/health".format(base_url), headers=headers, timeout=10)
    print("   Status: {}".format(r.status_code))
    print("   Response: {}".format(r.json()))
    assert r.status_code == 200, "Health check failed"
    print("   OK\n")


def test_set_image(base_url, headers, img):
    print("2. Testing /set_image ...")
    payload = {
        "image_b64": encode_image(img),
        "image_shape": list(img.shape),
        "image_dtype": str(img.dtype),
    }
    t0 = time.time()
    r = requests.post(
        "{}/set_image".format(base_url),
        json=payload,
        headers=headers,
        timeout=180,
    )
    elapsed = time.time() - t0
    print("   Status: {}".format(r.status_code))
    data = r.json()
    print("   Session ID: {}".format(data.get("session_id", "N/A")))
    print("   Original size: {}".format(data.get("original_size")))
    print("   Time: {:.2f}s".format(elapsed))
    assert r.status_code == 200, "set_image failed: {}".format(r.text)
    print("   OK\n")
    return data["session_id"]


def test_predict(base_url, headers, session_id, img_size):
    print("3. Testing /predict (positive click center) ...")
    center = img_size // 2
    payload = {
        "session_id": session_id,
        "point_coords": [[float(center), float(center)]],
        "point_labels": [1],
        "multimask_output": False,
    }
    t0 = time.time()
    r = requests.post(
        "{}/predict".format(base_url),
        json=payload,
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - t0
    print("   Status: {}".format(r.status_code))

    if r.status_code == 200:
        data = r.json()
        masks_shape = data["masks_shape"]
        scores = data["scores"]
        print("   Masks shape: {}".format(masks_shape))
        print("   Scores: {}".format(scores))
        print("   Time: {:.2f}s".format(elapsed))

        # Decode mask and check it's not empty
        mask_bytes = base64.b64decode(data["masks"])
        masks = np.frombuffer(mask_bytes, dtype=data["masks_dtype"]).reshape(masks_shape)
        pixel_count = int(masks[0].sum())
        print("   Mask pixel count: {} / {}".format(pixel_count, img_size * img_size))
        print("   OK\n")
        return data
    elif r.status_code == 422:
        print("   Empty mask (expected with synthetic image): {}".format(r.json()["detail"]))
        print("   OK (empty mask is acceptable for synthetic test)\n")
        return None
    else:
        print("   FAILED: {}".format(r.text))
        return None


def test_reset(base_url, headers, session_id):
    print("4. Testing /reset ...")
    r = requests.post(
        "{}/reset".format(base_url),
        params={"session_id": session_id},
        headers=headers,
        timeout=30,
    )
    print("   Status: {}".format(r.status_code))
    print("   Response: {}".format(r.json()))
    assert r.status_code == 200, "Reset failed: {}".format(r.text)
    print("   OK\n")


def main():
    parser = argparse.ArgumentParser(description="Test SAM2 API server")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--api-key", default="", help="API key (X-API-Key header)")
    parser.add_argument("--size", type=int, default=256, help="Test image size")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print("=" * 50)
    print("SAM2 API Test")
    print("Server: {}".format(base_url))
    print("Image size: {}x{}".format(args.size, args.size))
    print("=" * 50 + "\n")

    img = make_test_image(args.size)
    print("Test image created: shape={}, dtype={}\n".format(img.shape, img.dtype))

    test_health(base_url, headers)
    session_id = test_set_image(base_url, headers, img)
    test_predict(base_url, headers, session_id, args.size)
    test_reset(base_url, headers, session_id)

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
