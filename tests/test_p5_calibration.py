from __future__ import annotations

import copy

import numpy as np
import pytest

from rl_attack.defenses.rapid_guard.calibration import (
    CleanCalibrationCohort,
    RapidGuardArtifact,
    calibrate_split_conformal,
    finite_sample_order_index,
)
from rl_attack.defenses.rapid_guard.contracts import (
    CERTIFICATE_SCOPE,
    DetectorChannels,
    RapidGuardBinding,
    SplitSeedRegistry,
)
from rl_attack.defenses.rapid_guard.detector import (
    FusionFitCohort,
    FusionFitConfig,
    FusionTrainingResult,
    fit_attack_exposed_fusion,
)

HASHES = {
    letter: f"{index:064x}"
    for index, letter in enumerate("abcdefghijkl", start=1)
}


def binding(*, alpha: float = 0.1) -> RapidGuardBinding:
    return RapidGuardBinding(
        victim_checkpoint_sha256=HASHES["a"],
        victim_policy_state_sha256=HASHES["b"],
        environment_contract_sha256=HASHES["c"],
        observation_space_sha256=HASHES["d"],
        action_space_sha256=HASHES["e"],
        normalization_contract_sha256=HASHES["f"],
        projector_contract_sha256=HASHES["g"],
        certificate_epsilon=0.05,
        fit_dataset_sha256=HASHES["h"],
        calibration_dataset_sha256=HASHES["i"],
        attack_families=("p3_robust_sarsa", "p4_stfa"),
        seed=23,
        alpha=alpha,
    )


def registry() -> SplitSeedRegistry:
    return SplitSeedRegistry(
        fit=(1, 2),
        calibration=tuple(range(10, 30)),
        test=(30, 31),
    )


def detector_channels(
    temporal: np.ndarray,
    divergence: np.ndarray | None = None,
    deficit: np.ndarray | None = None,
) -> DetectorChannels:
    divergence = temporal * 0.3 if divergence is None else divergence
    deficit = temporal * 0.7 if deficit is None else deficit
    return DetectorChannels(
        np.asarray(temporal, dtype=np.float64),
        np.asarray(divergence, dtype=np.float64),
        np.asarray(deficit, dtype=np.float64),
    )


def trained_result() -> FusionTrainingResult:
    clean = np.linspace(0.01, 0.2, 20, dtype=np.float64)
    attacked = np.linspace(1.0, 1.2, 20, dtype=np.float64)
    values = np.concatenate([clean, attacked])
    cohort = FusionFitCohort(
        channels=detector_channels(values),
        attacked=np.asarray([False] * 20 + [True] * 20, dtype=np.bool_),
        attack_family=(
            *("clean" for _ in range(20)),
            *("p3_robust_sarsa" for _ in range(10)),
            *("p4_stfa" for _ in range(10)),
        ),
        episode_seeds=np.asarray([1, 2] * 20, dtype=np.int64),
        dataset_sha256=HASHES["h"],
    )
    return fit_attack_exposed_fusion(
        cohort,
        binding=binding(),
        split_registry=registry(),
        config=FusionFitConfig(
            gradient_steps=200,
            learning_rate=0.05,
            l2_penalty=0.001,
            scale_floor=1.0e-8,
        ),
    )


def calibration_cohort(*, n: int = 20) -> CleanCalibrationCohort:
    values = np.linspace(0.02, 0.32, n, dtype=np.float64)
    return CleanCalibrationCohort(
        channels=detector_channels(values),
        attacked=np.zeros(n, dtype=np.bool_),
        episode_seeds=np.arange(10, 10 + n, dtype=np.int64),
        dataset_sha256=HASHES["i"],
    )


def calibrated_artifact() -> RapidGuardArtifact:
    return calibrate_split_conformal(
        trained_result(),
        calibration_cohort(),
        binding=binding(),
        split_registry=registry(),
    )


@pytest.mark.parametrize(
    ("n", "alpha", "expected"),
    [
        (9, 0.1, 8),
        (19, 0.1, 17),
        (20, 0.1, 18),
        (20, 0.2, 16),
    ],
)
def test_finite_sample_order_index_matches_exact_formula(
    n: int,
    alpha: float,
    expected: int,
) -> None:
    assert finite_sample_order_index(n, alpha) == expected


def test_finite_sample_order_index_rejects_unsupported_alpha_and_integer_alpha() -> None:
    with pytest.raises(ValueError, match="too small"):
        finite_sample_order_index(9, 0.01)
    with pytest.raises(TypeError, match="floating"):
        finite_sample_order_index(20, 1)


def test_clean_split_conformal_threshold_is_exact_and_deterministic() -> None:
    training = trained_result()
    cohort = calibration_cohort()
    first = calibrate_split_conformal(
        training,
        cohort,
        binding=binding(),
        split_registry=registry(),
    )
    second = calibrate_split_conformal(
        training,
        cohort,
        binding=binding(),
        split_registry=registry(),
    )
    expected_scores = np.sort(training.head.score(cohort.channels), kind="stable")
    assert first.order_index == 18
    assert first.threshold == expected_scores[18]
    assert first.threshold == second.threshold
    np.testing.assert_array_equal(first.calibration_scores, second.calibration_scores)
    assert first.manifest == second.manifest
    assert not first.calibration_scores.flags.writeable
    assert first.manifest["calibration"]["unit"] == "clean_episode"
    assert (
        first.manifest["calibration"]["within_episode_aggregation"]
        == "maximum_risk"
    )


def test_calibration_aggregates_repeated_steps_to_one_episode_maximum() -> None:
    training = trained_result()
    episode_seeds = np.repeat(np.arange(10, 20, dtype=np.int64), 2)
    values = np.linspace(0.01, 0.4, episode_seeds.shape[0], dtype=np.float64)
    cohort = CleanCalibrationCohort(
        channels=detector_channels(values),
        attacked=np.zeros(values.shape[0], dtype=np.bool_),
        episode_seeds=episode_seeds,
        dataset_sha256=HASHES["i"],
    )
    artifact = calibrate_split_conformal(
        training,
        cohort,
        binding=binding(),
        split_registry=registry(),
    )
    per_step = training.head.score(cohort.channels)
    maxima = np.asarray(
        [np.max(per_step[episode_seeds == seed]) for seed in range(10, 20)],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        artifact.calibration_scores,
        np.sort(maxima, kind="stable"),
    )
    assert artifact.calibration_scores.shape == (10,)
    assert artifact.order_index == 9


def test_threshold_uses_strict_upper_tail_comparison() -> None:
    artifact = calibrated_artifact()
    scores = np.asarray(
        [
            artifact.threshold - 1.0e-6,
            artifact.threshold,
            artifact.threshold + 1.0e-6,
        ],
        dtype=np.float64,
    )
    decisions = artifact.is_anomalous(scores)
    np.testing.assert_array_equal(decisions, [False, False, True])
    assert not decisions.flags.writeable


def test_manifest_binds_scientific_provenance_ablation_and_claim_scope() -> None:
    artifact = calibrated_artifact()
    manifest = artifact.manifest
    binding_manifest = manifest["binding"]
    assert binding_manifest["victim_checkpoint_sha256"] == HASHES["a"]
    assert binding_manifest["victim_policy_state_sha256"] == HASHES["b"]
    assert binding_manifest["environment_contract_sha256"] == HASHES["c"]
    assert binding_manifest["observation_space_sha256"] == HASHES["d"]
    assert binding_manifest["action_space_sha256"] == HASHES["e"]
    assert binding_manifest["normalization_contract_sha256"] == HASHES["f"]
    assert binding_manifest["projector_contract_sha256"] == HASHES["g"]
    assert binding_manifest["fit"]["dataset_sha256"] == HASHES["h"]
    assert binding_manifest["calibration"]["dataset_sha256"] == HASHES["i"]
    assert binding_manifest["certificate"]["epsilon"] == 0.05
    assert binding_manifest["attack_families"] == [
        "p3_robust_sarsa",
        "p4_stfa",
    ]
    assert binding_manifest["seed"] == 23
    assert binding_manifest["calibration"]["alpha"] == 0.1
    assert manifest["detector"]["active_channels"] == [
        "temporal_innovation",
        "categorical_js",
        "ibp_margin_deficit",
    ]
    assert manifest["detector"]["ablation_is_explicit"] is True
    assert manifest["claims"]["ibp_certificate_scope"] == CERTIFICATE_SCOPE
    assert manifest["claims"]["certifies_episode_return"] is False
    assert manifest["claims"]["certifies_safety"] is False


def test_artifact_payload_round_trip_and_arrays_remain_immutable() -> None:
    artifact = calibrated_artifact()
    payload = artifact.to_payload()
    restored = RapidGuardArtifact.from_payload(payload)
    assert restored.manifest == artifact.manifest
    assert restored.threshold == artifact.threshold
    np.testing.assert_array_equal(restored.head.weights, artifact.head.weights)
    np.testing.assert_array_equal(
        restored.calibration_scores,
        artifact.calibration_scores,
    )
    payload["state"]["head"]["weights"][0] += 100.0
    np.testing.assert_array_equal(restored.head.weights, artifact.head.weights)
    assert not restored.head.weights.flags.writeable
    assert not restored.calibration_scores.flags.writeable


@pytest.mark.parametrize("tamper_kind", ["head", "scores", "binding", "claims", "extra"])
def test_artifact_payload_rejects_tampering(tamper_kind: str) -> None:
    payload = calibrated_artifact().to_payload()
    if tamper_kind == "head":
        payload["state"]["head"]["weights"][0] += 0.25
    elif tamper_kind == "scores":
        payload["state"]["calibration_scores"][0] += 1.0e-4
    elif tamper_kind == "binding":
        payload["manifest"]["binding"]["projector_contract_sha256"] = HASHES["j"]
    elif tamper_kind == "claims":
        payload["manifest"]["claims"]["certifies_safety"] = True
    else:
        payload["manifest"]["unexpected"] = "forbidden"
    with pytest.raises(ValueError, match="tamper|schema"):
        RapidGuardArtifact.from_payload(payload)


def test_payload_rejects_integer_array_encoding_instead_of_casting() -> None:
    payload = calibrated_artifact().to_payload()
    payload["state"]["calibration_scores"][0] = 0
    with pytest.raises(TypeError, match="floating-point list"):
        RapidGuardArtifact.from_payload(payload)


def test_calibration_cohort_rejects_attack_and_non_calibration_role() -> None:
    channels = detector_channels(np.asarray([0.1, 0.2], dtype=np.float64))
    with pytest.raises(ValueError, match="clean only"):
        CleanCalibrationCohort(
            channels=channels,
            attacked=np.asarray([False, True], dtype=np.bool_),
            episode_seeds=np.asarray([10, 11], dtype=np.int64),
            dataset_sha256=HASHES["i"],
        )
    with pytest.raises(ValueError, match="exactly 'calibration'"):
        CleanCalibrationCohort(
            channels=channels,
            attacked=np.zeros(2, dtype=np.bool_),
            episode_seeds=np.asarray([10, 11], dtype=np.int64),
            dataset_sha256=HASHES["i"],
            role="test",
        )


def test_calibration_rejects_reserved_test_seed_and_wrong_dataset() -> None:
    training = trained_result()
    clean = calibration_cohort()
    leaked = CleanCalibrationCohort(
        channels=clean.channels,
        attacked=clean.attacked,
        episode_seeds=np.full(clean.channels.n_samples, 30, dtype=np.int64),
        dataset_sha256=clean.dataset_sha256,
    )
    with pytest.raises(ValueError, match="outside|test seeds"):
        calibrate_split_conformal(
            training,
            leaked,
            binding=binding(),
            split_registry=registry(),
        )
    wrong_dataset = CleanCalibrationCohort(
        channels=clean.channels,
        attacked=clean.attacked,
        episode_seeds=clean.episode_seeds,
        dataset_sha256=HASHES["j"],
    )
    with pytest.raises(ValueError, match="dataset hash"):
        calibrate_split_conformal(
            training,
            wrong_dataset,
            binding=binding(),
            split_registry=registry(),
        )


def test_calibration_rejects_forged_fit_evidence_with_test_seed() -> None:
    training = trained_result()
    forged = FusionTrainingResult(
        head=training.head,
        fit_episode_seeds=(30,),
        initial_loss=training.initial_loss,
        final_loss=training.final_loss,
        observed_attack_families=training.observed_attack_families,
        class_counts=training.class_counts,
    )
    with pytest.raises(ValueError, match="non-fit|test"):
        calibrate_split_conformal(
            forged,
            calibration_cohort(),
            binding=binding(),
            split_registry=registry(),
        )


def test_artifact_constructor_rejects_wrong_threshold_and_order_index() -> None:
    artifact = calibrated_artifact()
    with pytest.raises(ValueError, match="threshold"):
        RapidGuardArtifact(
            head=artifact.head,
            binding=artifact.binding,
            split_registry=artifact.split_registry,
            fit_episode_seeds=artifact.fit_episode_seeds,
            calibration_episode_seeds=artifact.calibration_episode_seeds,
            threshold=artifact.threshold - 1.0e-4,
            calibration_scores=artifact.calibration_scores,
            order_index=artifact.order_index,
        )
    with pytest.raises(ValueError, match="order_index"):
        RapidGuardArtifact(
            head=artifact.head,
            binding=artifact.binding,
            split_registry=artifact.split_registry,
            fit_episode_seeds=artifact.fit_episode_seeds,
            calibration_episode_seeds=artifact.calibration_episode_seeds,
            threshold=artifact.threshold,
            calibration_scores=artifact.calibration_scores,
            order_index=artifact.order_index - 1,
        )


def test_payload_schema_is_exact() -> None:
    payload = calibrated_artifact().to_payload()
    extra = copy.deepcopy(payload)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        RapidGuardArtifact.from_payload(extra)
