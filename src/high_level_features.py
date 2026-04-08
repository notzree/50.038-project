from __future__ import annotations

from typing import Mapping


HIGH_LEVEL_FEATURE_COLUMNS = [
    "danceability_proxy",
    "energy_proxy",
    "acousticness_proxy",
    "instrumentalness_proxy",
    "speechiness_proxy",
    "valence_proxy",
    "brightness_proxy",
    "rhythmic_stability_proxy",
    "high_level_features_version",
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    x = (value - min_v) / (max_v - min_v)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def compute_high_level_features(row: Mapping[str, object]) -> dict[str, float | str]:
    tempo = _safe_float(row.get("tempo_bpm"), 0.0)
    rms = _safe_float(row.get("rms_mean"), 0.0)
    centroid = _safe_float(row.get("spectral_centroid_mean"), 0.0)
    rolloff = _safe_float(row.get("spectral_rolloff_mean"), 0.0)
    zcr = _safe_float(row.get("zcr_mean"), 0.0)

    onset_mean = _safe_float(row.get("onset_strength_mean"), 0.0)
    tempo_stability = _safe_float(row.get("tempo_stability"), 0.0)
    mfcc_1 = _safe_float(row.get("mfcc_1_mean"), 0.0)
    mfcc_2 = _safe_float(row.get("mfcc_2_mean"), 0.0)
    mfcc_3 = _safe_float(row.get("mfcc_3_mean"), 0.0)
    bass_ratio = _safe_float(row.get("bass_energy_ratio"), 0.0)

    tempo_n = _norm(tempo, 60.0, 180.0)
    rms_n = _norm(rms, 0.02, 0.25)
    centroid_n = _norm(centroid, 500.0, 5000.0)
    rolloff_n = _norm(rolloff, 1500.0, 7000.0)
    zcr_n = _norm(zcr, 0.02, 0.25)
    onset_n = _norm(onset_mean, 0.5, 12.0)
    stability_n = 1.0 - _norm(tempo_stability, 0.0, 0.35)
    bass_n = _norm(bass_ratio, 0.05, 0.45)

    danceability_existing = row.get("danceability_score")
    energy_existing = row.get("energy_score")

    danceability_proxy = (
        _safe_float(danceability_existing, 0.0)
        if danceability_existing is not None
        else (0.35 * tempo_n + 0.35 * onset_n + 0.30 * stability_n)
    )
    energy_proxy = (
        _safe_float(energy_existing, 0.0)
        if energy_existing is not None
        else (0.45 * rms_n + 0.30 * centroid_n + 0.25 * rolloff_n)
    )

    brightness_proxy = 0.55 * centroid_n + 0.45 * rolloff_n
    rhythmic_stability_proxy = stability_n

    acousticness_proxy = 1.0 - (
        0.45 * zcr_n + 0.30 * brightness_proxy + 0.25 * energy_proxy
    )
    acousticness_proxy = _norm(acousticness_proxy, 0.0, 1.0)

    speechiness_proxy = _norm(
        0.55 * zcr_n + 0.45 * _norm(mfcc_2, -120.0, 60.0), 0.0, 1.0
    )

    instrumentalness_proxy = _norm(
        0.45 * acousticness_proxy + 0.35 * bass_n + 0.20 * (1.0 - speechiness_proxy),
        0.0,
        1.0,
    )

    valence_proxy = _norm(
        0.30 * tempo_n
        + 0.25 * brightness_proxy
        + 0.25 * _norm(mfcc_1, -500.0, 100.0)
        + 0.20 * _norm(mfcc_3, -120.0, 120.0)
        - 0.15 * _norm(abs(mfcc_2), 0.0, 120.0),
        0.0,
        1.0,
    )

    return {
        "danceability_proxy": _norm(danceability_proxy, 0.0, 1.0),
        "energy_proxy": _norm(energy_proxy, 0.0, 1.0),
        "acousticness_proxy": acousticness_proxy,
        "instrumentalness_proxy": instrumentalness_proxy,
        "speechiness_proxy": speechiness_proxy,
        "valence_proxy": valence_proxy,
        "brightness_proxy": _norm(brightness_proxy, 0.0, 1.0),
        "rhythmic_stability_proxy": _norm(rhythmic_stability_proxy, 0.0, 1.0),
        "high_level_features_version": "v1",
    }
