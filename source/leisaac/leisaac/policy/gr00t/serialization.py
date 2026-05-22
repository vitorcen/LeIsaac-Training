import io
import json

import numpy as np

try:
    import msgpack
    import msgpack_numpy as mnp
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "msgpack, msgpack-numpy, pydantic are required: "
        "pip install msgpack msgpack-numpy pydantic"
    )


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class MsgSerializer:
    """Wire serializer for the LeIsaac ⇄ GR00T policy server family.

    Output (encoding) uses the legacy ``__ndarray_class__`` format that all of
    Isaac-GR00T's N1.5, old N1.6, and the patched N1.7 server (see
    ``dependencies/Isaac-GR00T/gr00t/policy/server_client.py``) accept.

    Input (decoding) accepts BOTH formats:
      - legacy ``__ndarray_class__`` (np.save into BytesIO)
      - msgpack-numpy ``{nd: True, type, shape, data}`` produced by N1.7
        servers that return ndarrays without going through our encoder
    so the client can talk to any server in the family.
    """

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode_custom)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode_custom, raw=False)

    @staticmethod
    def _encode_custom(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj

    @staticmethod
    def _decode_custom(obj):
        if not isinstance(obj, dict):
            return obj
        # Server may key fields as bytes or str depending on msgpack version;
        # treat both.
        keys = set(obj.keys())
        if "__ModalityConfig_class__" in keys or b"__ModalityConfig_class__" in keys:
            key = "as_json" if "as_json" in obj else b"as_json"
            payload = obj[key]
            if isinstance(payload, bytes):
                payload = payload.decode()
            return ModalityConfig(**json.loads(payload))
        if "__ndarray_class__" in keys or b"__ndarray_class__" in keys:
            key = "as_npy" if "as_npy" in obj else b"as_npy"
            return np.load(io.BytesIO(obj[key]), allow_pickle=False)
        # msgpack-numpy {nd:True, ...} format (N1.7 server returns this for
        # ndarrays unless the server is patched to use __ndarray_class__).
        if obj.get("nd") is True:
            return mnp.decode(obj)
        return obj
