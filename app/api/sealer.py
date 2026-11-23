"""Optional app-level encryption for what the job store persists (uploads, results, file names, caller metadata).

With RESULT_ENCRYPTION_KEY (base64 of 32 random bytes) set, blobs are AES-256-GCM encrypted, bound to their job id,
so database dumps, backups and WAL never hold document content in plaintext. Without a key they are stored as is.

    uv run python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"
"""

import base64
import binascii
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PLAIN, AES_GCM = b"\x00", b"\x01"
NONCE = 12


class SealError(RuntimeError):
    pass


class Sealer:
    def __init__(self, key_b64: str = ""):
        self._aes = None
        if key_b64:
            try:
                key = base64.b64decode(key_b64, validate=True)
            except binascii.Error as e:
                raise SealError("RESULT_ENCRYPTION_KEY is not valid base64") from e
            if len(key) != 32:
                raise SealError("RESULT_ENCRYPTION_KEY must decode to 32 bytes")
            self._aes = AESGCM(key)

    @property
    def encrypting(self) -> bool:
        return self._aes is not None

    def seal(self, data: bytes, aad: str) -> bytes:
        if self._aes is None:
            return PLAIN + data
        nonce = os.urandom(NONCE)
        return AES_GCM + nonce + self._aes.encrypt(nonce, data, aad.encode())

    def open(self, blob: bytes, aad: str) -> bytes:
        blob = bytes(blob)
        kind, rest = blob[:1], blob[1:]
        if kind == PLAIN:
            return rest
        if kind == AES_GCM:
            if self._aes is None:
                raise SealError("stored data is encrypted but RESULT_ENCRYPTION_KEY is not set")
            try:
                return self._aes.decrypt(rest[:NONCE], rest[NONCE:], aad.encode())
            except Exception as e:  # noqa: BLE001 - wrong key or tampered data
                raise SealError("stored data could not be decrypted (wrong RESULT_ENCRYPTION_KEY?)") from e
        raise SealError("unknown stored data format")
