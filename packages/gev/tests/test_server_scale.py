"""The runtime LoRA-scale endpoint: off unless asked for, and bounded."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from gev import server  # noqa: E402


class FakeEngine:
    lora_scale = 1.0
    checkpoint = None

    def __init__(self, adapter: bool = True):
        self.adapter = adapter

    def set_lora_scale(self, scale: float) -> None:
        if not self.adapter:
            raise ValueError("no LoRA adapter is loaded to scale")
        self.lora_scale = scale


@pytest.fixture
def make_client(monkeypatch):
    def make(allow: bool, adapter: bool = True):
        monkeypatch.setattr(server, "load", lambda *args, **kwargs: FakeEngine(adapter))
        return TestClient(server.create_app(allow_lora_scale=allow))

    return make


def test_the_endpoint_does_not_exist_unless_enabled(make_client):
    with make_client(allow=False) as client:
        assert client.post("/admin/lora-scale", json={"scale": 0.5}).status_code == 404


def test_the_scale_is_applied_and_reported(make_client):
    with make_client(allow=True) as client:
        response = client.post("/admin/lora-scale", json={"scale": 0.5})
        assert response.status_code == 200 and response.json() == {"lora_scale": 0.5}


@pytest.mark.parametrize("scale", [-0.1, 2.5])
def test_a_scale_outside_the_range_is_rejected(make_client, scale):
    with make_client(allow=True) as client:
        assert client.post("/admin/lora-scale", json={"scale": scale}).status_code == 422


def test_scaling_a_model_without_an_adapter_is_a_conflict(make_client):
    with make_client(allow=True, adapter=False) as client:
        response = client.post("/admin/lora-scale", json={"scale": 0.5})
        assert response.status_code == 409 and "no LoRA adapter" in response.json()["detail"]
