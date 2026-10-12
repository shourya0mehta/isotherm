"""Kalshi API key loading and request signing.

Kalshi signs every authenticated request with the pre-sign text
    f"{timestamp_ms}{METHOD}{path_without_query}"
using Ed25519 (recommended) or RSA-PSS/SHA-256, base64-encoded.
The path must include the /trade-api/v2 prefix and must NOT include the query string.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


def load_private_key(pem: bytes):
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, (Ed25519PrivateKey, RSAPrivateKey)):
        raise ValueError("Kalshi keys must be Ed25519 or RSA")
    return key


def sign_text(private_key, text: str) -> str:
    msg = text.encode("utf-8")
    if isinstance(private_key, Ed25519PrivateKey):
        sig = private_key.sign(msg)
    else:
        sig = private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    return base64.b64encode(sig).decode("ascii")


@dataclass
class KalshiAuth:
    key_id: str
    private_key: object

    @classmethod
    def from_env(cls) -> Optional["KalshiAuth"]:
        """KALSHI_KEY_ID plus either KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY (PEM text)."""
        key_id = os.environ.get("KALSHI_KEY_ID", "").strip()
        if not key_id:
            return None
        path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if path:
            with open(os.path.expanduser(path), "rb") as f:
                pem = f.read()
        else:
            pem_txt = os.environ.get("KALSHI_PRIVATE_KEY", "")
            if not pem_txt:
                return None
            pem = pem_txt.replace("\\n", "\n").encode()
        return cls(key_id=key_id, private_key=load_private_key(pem))

    def headers(self, method: str, path: str, ts_ms: Optional[int] = None) -> dict:
        ts = str(ts_ms if ts_ms is not None else int(time.time() * 1000))
        path_no_q = path.split("?", 1)[0]
        sig = sign_text(self.private_key, ts + method.upper() + path_no_q)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }
