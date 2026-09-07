import torch

from crowdmon.device import device_name, resolve_device


def test_cpu_always_available():
    dev = resolve_device("cpu")
    assert dev.type == "cpu"


def test_auto_resolves_to_a_real_device():
    dev = resolve_device("auto")
    assert dev.type in ("cpu", "cuda", "mps")


def test_unavailable_backend_falls_back_to_cpu():
    # This container has no GPU, so explicitly requesting cuda/mps must not crash.
    dev = resolve_device("cuda")
    assert dev.type == "cpu"
    dev = resolve_device("mps")
    assert dev.type == "cpu"


def test_device_name_readable():
    assert isinstance(device_name(torch.device("cpu")), str)
