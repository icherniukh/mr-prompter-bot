import argparse
import json
from pathlib import Path

import pytest

from scripts import evaluate_upscaling_quality as harness


def test_parse_factors_accepts_two_or_three_k_values():
    assert harness.parse_factors("2,3") == [2.0, 3.0]
    assert harness.parse_factors("2,3,4") == [2.0, 3.0, 4.0]


@pytest.mark.parametrize("value", ["2", "2,3,4,5", "1,2", "two,3"])
def test_parse_factors_rejects_invalid_k_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        harness.parse_factors(value)


def test_local_only_run_writes_outputs_manifest_and_skips_gemini(tmp_path, monkeypatch):
    source = tmp_path / "sample.png"
    source.write_bytes(b"fake image input")
    calls = []

    def fake_upscale(
        input_path: Path,
        output_path: Path,
        factor: float,
        resampler: str,
        output_format: str = "jpeg",
    ) -> None:
        calls.append((input_path, output_path, factor, resampler, output_format))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{factor}:{resampler}".encode())

    monkeypatch.setattr(harness, "upscale_local_image", fake_upscale)

    manifest = harness.run_evaluation(
        [source],
        tmp_path / "eval",
        factors=[2.0, 3.0],
        resamplers=["nearest"],
        partial_factors=[1.5, 2.0],
        output_format="jpeg",
        enable_gemini=False,
        api_key=None,
        model="test-model",
    )

    manifest_path = tmp_path / "eval" / "manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded == manifest

    statuses_by_strategy = {}
    for entry in manifest["entries"]:
        statuses_by_strategy.setdefault(entry["strategy"], set()).add(entry["status"])

    assert statuses_by_strategy["manual"] == {"ok"}
    assert statuses_by_strategy["manual-gemini-repair"] == {"skipped"}
    assert statuses_by_strategy["partial-gemini-finish"] == {"skipped"}
    assert all(entry["output_path"] for entry in manifest["entries"] if entry["strategy"] == "manual")
    assert all(
        entry["output_path"].endswith(".jpg")
        for entry in manifest["entries"]
        if entry["strategy"] == "manual"
    )
    assert manifest["config"]["output_format"] == "jpeg"
    assert len(calls) == 5
    assert {call[-1] for call in calls} == {"jpeg"}


def test_gemini_strategy_requires_flag_and_api_key(tmp_path):
    entries = []
    harness.run_gemini_strategy(
        entries,
        input_path=tmp_path / "input.png",
        output_stem=tmp_path / "output",
        source=tmp_path / "source.png",
        strategy="manual-gemini-repair",
        target_factor=2.0,
        local_factor=2.0,
        resampler="lanczos",
        prompt="repair",
        enable_gemini=True,
        api_key=None,
        model="test-model",
    )

    assert entries[0].status == "skipped"
    assert entries[0].error == "No GEMINI_API_KEY or GOOGLE_API_KEY set"
