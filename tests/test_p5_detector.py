from __future__ import annotations

import copy

import numpy as np
import pytest

from rl_attack.defenses.rapid_guard.contracts import (
    CERTIFICATE_SCOPE,
    DETECTOR_CHANNELS,
    DetectorChannels,
    RapidGuardBinding,
    SplitSeedRegistry,
)
from rl_attack.defenses.rapid_guard.detector import (
    FrozenLogisticRiskHead,
    FusionFitCohort,
    FusionFitConfig,
    categorical_js_divergence,
    evaluate_detector_channels,
    fit_attack_exposed_fusion,
    ibp_greedy_action_margin_deficit,
    temporal_innovation_score,
)

HASHES = {
    letter: f"{index:064x}"
    for index, letter in enumerate("abcdefghijkl", start=1)
}


def binding() -> RapidGuardBinding:
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
        attack_families=("p3_pa_ad", "p4_stfa"),
        seed=17,
        alpha=0.1,
    )


def split_registry() -> SplitSeedRegistry:
    return SplitSeedRegistry(
        fit=(1, 2),
        calibration=(10, 11),
        test=(20, 21),
    )


def separable_fit_cohort() -> FusionFitCohort:
    clean_count = 20
    attacked_count = 20
    offset = np.linspace(0.0, 0.08, clean_count, dtype=np.float64)
    temporal = np.concatenate([0.05 + offset, 1.1 + offset])
    divergence = np.concatenate([0.01 + 0.2 * offset, 0.45 + 0.2 * offset])
    deficit = np.concatenate([0.02 + 0.1 * offset, 0.9 + 0.1 * offset])
    return FusionFitCohort(
        channels=DetectorChannels(temporal, divergence, deficit),
        attacked=np.asarray(
            [False] * clean_count + [True] * attacked_count,
            dtype=np.bool_,
        ),
        attack_family=(
            *("clean" for _ in range(clean_count)),
            *("p3_pa_ad" for _ in range(attacked_count // 2)),
            *("p4_stfa" for _ in range(attacked_count // 2)),
        ),
        episode_seeds=np.asarray(
            [1, 2] * ((clean_count + attacked_count) // 2),
            dtype=np.int64,
        ),
        dataset_sha256=HASHES["h"],
    )


def test_temporal_innovation_matches_constant_velocity_equation() -> None:
    history = np.asarray(
        [
            [[0.0, 2.0], [1.0, 4.0], [2.0, 6.0]],
            [[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]],
        ],
        dtype=np.float64,
    )
    result = temporal_innovation_score(
        history,
        innovation_scale=np.asarray([2.0, 1.0], dtype=np.float64),
    )
    np.testing.assert_allclose(result, np.asarray([0.0, np.sqrt(2.125)]))
    assert not result.flags.writeable


@pytest.mark.parametrize(
    ("history", "scale", "error"),
    [
        (np.zeros((2, 3, 2), dtype=np.int64), 1.0, TypeError),
        (np.zeros((2, 2, 2), dtype=np.float64), 1.0, ValueError),
        (np.zeros((2, 3, 2), dtype=np.float64), 0.0, ValueError),
        (
            np.asarray(
                [[[0.0], [1.0], [float("nan")]]],
                dtype=np.float64,
            ),
            1.0,
            ValueError,
        ),
    ],
)
def test_temporal_innovation_rejects_invalid_contracts(
    history: np.ndarray,
    scale: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        temporal_innovation_score(history, innovation_scale=scale)


def test_categorical_js_has_exact_endpoints_and_symmetry() -> None:
    left = np.asarray([[1.0, 0.0], [0.25, 0.75]], dtype=np.float64)
    right = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float64)
    forward = categorical_js_divergence(left, right)
    backward = categorical_js_divergence(right, left)
    np.testing.assert_allclose(forward, np.asarray([np.log(2.0), 0.0]))
    np.testing.assert_allclose(forward, backward)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[1, 0]], dtype=np.int64),
        np.asarray([[0.2, 0.2]], dtype=np.float64),
        np.asarray([[1.1, -0.1]], dtype=np.float64),
        np.asarray([[float("nan"), 0.0]], dtype=np.float64),
    ],
)
def test_categorical_js_rejects_invalid_probability_rows(
    probabilities: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        categorical_js_divergence(
            probabilities,
            np.asarray([[0.5, 0.5]], dtype=np.float64),
        )


def test_ibp_margin_deficit_uses_clean_greedy_action_and_narrow_claim() -> None:
    clean = np.asarray([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    lower = np.asarray([[1.5, 0.5, -0.2], [-0.2, 1.0, 0.2]], dtype=np.float64)
    upper = np.asarray([[2.5, 1.2, 0.4], [1.4, 4.0, 1.2]], dtype=np.float64)
    result = ibp_greedy_action_margin_deficit(
        clean,
        lower,
        upper,
        required_margin=0.2,
    )
    np.testing.assert_array_equal(result.clean_greedy_action, [0, 1])
    np.testing.assert_allclose(result.certified_margin, [0.3, -0.4])
    np.testing.assert_allclose(result.deficit, [0.0, 0.6])
    np.testing.assert_array_equal(result.action_invariant, [True, False])
    assert result.certificate_scope == CERTIFICATE_SCOPE
    assert result.certifies_episode_return is False
    assert result.certifies_safety is False


def test_ibp_margin_deficit_rejects_shape_bounds_and_non_float_inputs() -> None:
    clean = np.zeros((2, 2), dtype=np.float64)
    lower = np.zeros((2, 2), dtype=np.float64)
    upper = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="equal shape"):
        ibp_greedy_action_margin_deficit(clean, lower[:1], upper)
    bad_lower = lower.copy()
    bad_lower[0, 0] = 2.0
    with pytest.raises(ValueError, match="lower logit"):
        ibp_greedy_action_margin_deficit(clean, bad_lower, upper)
    with pytest.raises(TypeError, match="floating"):
        ibp_greedy_action_margin_deficit(clean.astype(np.int64), lower, upper)


def test_evaluate_channels_aligns_three_signals() -> None:
    result = evaluate_detector_channels(
        observation_history=np.asarray(
            [
                [[0.0], [1.0], [2.0]],
                [[0.0], [1.0], [3.0]],
            ],
            dtype=np.float64,
        ),
        innovation_scale=1.0,
        current_action_probabilities=np.asarray(
            [[0.8, 0.2], [0.2, 0.8]],
            dtype=np.float64,
        ),
        reference_action_probabilities=np.asarray(
            [[0.8, 0.2], [0.8, 0.2]],
            dtype=np.float64,
        ),
        clean_logits=np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        ibp_lower_logits=np.asarray([[1.5, -0.2], [-0.2, 0.4]], dtype=np.float64),
        ibp_upper_logits=np.asarray([[2.5, 0.5], [1.0, 2.5]], dtype=np.float64),
    )
    assert result.n_samples == 2
    np.testing.assert_allclose(result.temporal_innovation, [0.0, 1.0])
    np.testing.assert_allclose(result.ibp_margin_deficit, [0.0, 0.6])


def test_detector_channels_are_immutable_and_ablation_is_auditable() -> None:
    original = np.asarray([0.1, 0.2], dtype=np.float64)
    channels = DetectorChannels(original, original + 0.1, original + 0.2)
    original[0] = 99.0
    assert channels.temporal_innovation[0] == 0.1
    assert not channels.temporal_innovation.flags.writeable
    matrix = channels.matrix(("categorical_js", "ibp_margin_deficit"))
    np.testing.assert_allclose(matrix[:, 0], channels.categorical_js)
    assert not matrix.flags.writeable
    with pytest.raises(ValueError, match="canonical"):
        channels.matrix(("ibp_margin_deficit", "categorical_js"))
    with pytest.raises(ValueError, match="unknown"):
        channels.matrix(("not_a_channel",))
    with pytest.raises(ValueError, match=r"log\(2\)"):
        DetectorChannels(
            np.asarray([0.1], dtype=np.float64),
            np.asarray([0.8], dtype=np.float64),
            np.asarray([0.1], dtype=np.float64),
        )


def test_binding_round_trip_binds_scopes_roles_and_all_hashes() -> None:
    expected = binding()
    manifest = expected.to_manifest()
    actual = RapidGuardBinding.from_manifest(manifest)
    assert actual == expected
    assert manifest["fit"]["role"] == "fit"
    assert manifest["calibration"]["role"] == "calibration"
    assert manifest["certificate"]["scope"] == CERTIFICATE_SCOPE
    assert manifest["certificate"]["certifies_episode_return"] is False
    assert manifest["certificate"]["certifies_safety"] is False
    tampered = copy.deepcopy(manifest)
    tampered["certificate"]["certifies_safety"] = True
    with pytest.raises(ValueError, match="safety certificate"):
        RapidGuardBinding.from_manifest(tampered)


def test_binding_requires_strict_floats_and_p3_p4_coverage() -> None:
    values = binding().__dict__.copy()
    values["certificate_epsilon"] = 1
    with pytest.raises(TypeError, match="floating"):
        RapidGuardBinding(**values)
    values = binding().__dict__.copy()
    values["attack_families"] = ("p3_pa_ad",)
    with pytest.raises(ValueError, match="P3 and one P4"):
        RapidGuardBinding(**values)


def test_split_registry_rejects_overlap_unsorted_and_float_seeds() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        SplitSeedRegistry(fit=(1,), calibration=(2,), test=(1,))
    with pytest.raises(ValueError, match="sorted"):
        SplitSeedRegistry(fit=(2, 1), calibration=(3,), test=(4,))
    with pytest.raises(TypeError, match="integer"):
        SplitSeedRegistry(fit=(1.0,), calibration=(3,), test=(4,))


def test_attack_exposed_fusion_is_deterministic_and_separates_cohort() -> None:
    cohort = separable_fit_cohort()
    config = FusionFitConfig(
        gradient_steps=300,
        learning_rate=0.05,
        l2_penalty=0.001,
        scale_floor=1.0e-8,
    )
    first = fit_attack_exposed_fusion(
        cohort,
        binding=binding(),
        split_registry=split_registry(),
        config=config,
    )
    second = fit_attack_exposed_fusion(
        cohort,
        binding=binding(),
        split_registry=split_registry(),
        config=config,
    )
    np.testing.assert_array_equal(first.head.weights, second.head.weights)
    assert first.head.bias == second.head.bias
    assert first.head.state_sha256 == second.head.state_sha256
    assert first.final_loss < first.initial_loss
    risks = first.head.score(cohort.channels)
    assert risks[20:].min() > risks[:20].max()
    assert not first.head.weights.flags.writeable
    assert not risks.flags.writeable
    assert first.observed_attack_families == ("p3_pa_ad", "p4_stfa")


def test_fusion_ablation_is_bound_into_head_state() -> None:
    result = fit_attack_exposed_fusion(
        separable_fit_cohort(),
        binding=binding(),
        split_registry=split_registry(),
        active_channels=("categorical_js",),
        config=FusionFitConfig(
            gradient_steps=20,
            learning_rate=0.05,
            l2_penalty=0.0,
            scale_floor=1.0e-8,
        ),
    )
    assert result.head.active_channels == ("categorical_js",)
    assert result.head.weights.shape == (1,)
    restored = FrozenLogisticRiskHead.from_state(result.head.to_state())
    assert restored.state_sha256 == result.head.state_sha256


def test_fusion_rejects_role_seed_dataset_and_attack_family_leakage() -> None:
    cohort = separable_fit_cohort()
    common = {
        "binding": binding(),
        "split_registry": split_registry(),
        "config": FusionFitConfig(
            gradient_steps=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            scale_floor=1.0e-8,
        ),
    }
    with pytest.raises(ValueError, match="exactly 'fit'"):
        FusionFitCohort(
            channels=cohort.channels,
            attacked=cohort.attacked,
            attack_family=cohort.attack_family,
            episode_seeds=cohort.episode_seeds,
            dataset_sha256=cohort.dataset_sha256,
            role="test",
        )
    leaked = FusionFitCohort(
        channels=cohort.channels,
        attacked=cohort.attacked,
        attack_family=cohort.attack_family,
        episode_seeds=np.full(cohort.channels.n_samples, 20, dtype=np.int64),
        dataset_sha256=cohort.dataset_sha256,
    )
    with pytest.raises(ValueError, match="outside"):
        fit_attack_exposed_fusion(leaked, **common)
    wrong_dataset = FusionFitCohort(
        channels=cohort.channels,
        attacked=cohort.attacked,
        attack_family=cohort.attack_family,
        episode_seeds=cohort.episode_seeds,
        dataset_sha256=HASHES["j"],
    )
    with pytest.raises(ValueError, match="dataset hash"):
        fit_attack_exposed_fusion(wrong_dataset, **common)
    missing_p4_families = tuple(
        "p3_pa_ad" if attacked else "clean"
        for attacked in cohort.attacked.tolist()
    )
    missing_p4 = FusionFitCohort(
        channels=cohort.channels,
        attacked=cohort.attacked,
        attack_family=missing_p4_families,
        episode_seeds=cohort.episode_seeds,
        dataset_sha256=cohort.dataset_sha256,
    )
    with pytest.raises(ValueError, match="p4_stfa"):
        fit_attack_exposed_fusion(missing_p4, **common)


def test_fusion_fit_cohort_rejects_fractional_labels_and_bad_clean_family() -> None:
    cohort = separable_fit_cohort()
    with pytest.raises(TypeError, match="boolean"):
        FusionFitCohort(
            channels=cohort.channels,
            attacked=cohort.attacked.astype(np.float64),
            attack_family=cohort.attack_family,
            episode_seeds=cohort.episode_seeds,
            dataset_sha256=cohort.dataset_sha256,
        )
    families = list(cohort.attack_family)
    families[0] = "p3_pa_ad"
    with pytest.raises(ValueError, match="clean samples"):
        FusionFitCohort(
            channels=cohort.channels,
            attacked=cohort.attacked,
            attack_family=tuple(families),
            episode_seeds=cohort.episode_seeds,
            dataset_sha256=cohort.dataset_sha256,
        )


def test_public_channel_order_is_stable() -> None:
    assert DETECTOR_CHANNELS == (
        "temporal_innovation",
        "categorical_js",
        "ibp_margin_deficit",
    )
