import pickle
import struct

import numpy as np


NDARRAY_MARKER = "__numpy_ndarray__"
TUPLE_MARKER = "__python_tuple__"


def _encode(obj):
    """
    Convert numpy objects into pure Python + bytes objects
    before pickle serialization.

    This avoids cross-environment NumPy pickle incompatibilities.
    """

    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)

        return {
            NDARRAY_MARKER: True,
            "dtype": arr.dtype.str,
            "shape": tuple(arr.shape),
            "data": arr.tobytes(),
        }

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            key: _encode(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [_encode(value) for value in obj]

    if isinstance(obj, tuple):
        return {
            TUPLE_MARKER: [
                _encode(value)
                for value in obj
            ]
        }

    return obj


def _decode(obj):
    if isinstance(obj, dict):

        if obj.get(NDARRAY_MARKER, False):
            arr = np.frombuffer(
                obj["data"],
                dtype=np.dtype(obj["dtype"]),
            )

            # copy() makes the resulting ndarray writable
            return arr.reshape(
                obj["shape"]
            ).copy()

        if TUPLE_MARKER in obj:
            return tuple(
                _decode(value)
                for value in obj[TUPLE_MARKER]
            )

        return {
            key: _decode(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [_decode(value) for value in obj]

    return obj


def send_obj(sock, obj):
    safe_obj = _encode(obj)

    payload = pickle.dumps(
        safe_obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    sock.sendall(
        struct.pack("!Q", len(payload))
    )

    sock.sendall(payload)


def recv_exact(sock, n):
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError(
                "Peer disconnected"
            )

        data.extend(chunk)

    return bytes(data)


def recv_obj(sock):
    header = recv_exact(sock, 8)

    size = struct.unpack(
        "!Q",
        header,
    )[0]

    payload = recv_exact(sock, size)

    safe_obj = pickle.loads(payload)

    return _decode(safe_obj)
