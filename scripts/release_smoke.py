#!/usr/bin/env python3
"""Check the packaged dylib through its public C ABI on the actual Metal GPU."""
import ctypes as c
import math
import sys


class Status(c.Structure):
    _fields_ = [("code", c.c_int32), ("message", c.c_char * 256)]


def check(status):
    if status.code:
        raise SystemExit(f"native code={status.code}: {status.message.decode()}")


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    lib = c.CDLL(sys.argv[1])
    lib.welvet_version.restype = c.c_char_p
    require(lib.welvet_version().decode() == sys.argv[2], "dylib version mismatch")
    lib.welvet_model_abi_version.restype = c.c_int32
    require(lib.welvet_model_abi_version() == 1, "model ABI mismatch")
    for symbol in ("welvet_model_mount", "welvet_model_forward", "welvet_model_train_mse",
                   "welvet_model_export_weights", "welvet_model_sync", "welvet_model_release",
                   "welvet_metal_blob_free", "welvet_model_forward_pair", "welvet_model_backward",
                   "welvet_model_apply_sgd", "welvet_model_train_credit", "welvet_model_dense_proxy"):
        getattr(lib, symbol)
    lib.welvet_metal_init.restype = Status
    check(lib.welvet_metal_init())
    lib.welvet_runtime_identity.restype = c.c_char_p
    identity = lib.welvet_runtime_identity().decode()
    require(identity.startswith("metal:"), f"expected Metal identity: {identity}")
    fptr, dptr = c.POINTER(c.c_float), c.POINTER(c.c_double)
    geometry = [c.c_int32] * 6
    prefix = geometry + [fptr, dptr, c.c_size_t, fptr]
    lib.welvet_dense_forward_f32.argtypes = prefix + [fptr, fptr]
    lib.welvet_dense_forward_f32.restype = Status
    lib.welvet_dense_normalbp_f32.argtypes = prefix + [fptr, c.c_int32, c.c_double] + [fptr] * 5 + [dptr]
    lib.welvet_dense_normalbp_f32.restype = Status
    weights = (c.c_float * 4)(1, 0, 0, 1)
    x, target = (c.c_float * 2)(.5, -.25), (c.c_float * 2)(0, 0)
    pre, post, dx = (c.c_float * 2)(), (c.c_float * 2)(), (c.c_float * 2)()
    dw, after, loss = (c.c_float * 4)(), (c.c_float * 4)(), c.c_double()
    args = (1, 2, 2, -1, 1, 0, weights, None, 0, x)
    check(lib.welvet_dense_forward_f32(*args, pre, post))
    require(list(post) == [.5, -.25], "forward output mismatch")
    check(lib.welvet_dense_normalbp_f32(*args, target, 1, .1, pre, post, dx, dw, after, c.byref(loss)))
    require(abs(loss.value - .15625) < 1e-6, f"MSE mismatch: {loss.value}")
    expected = [.975, .0125, .0125, .99375]
    require(all(math.isfinite(a) and abs(a - b) < 1e-6 for a, b in zip(after, expected)), "SGD mismatch")
    print(f"Packaged GPU check passed: {identity}; forward, backward, SGD, ABI 1 + segment extension")


if __name__ == "__main__":
    main()
