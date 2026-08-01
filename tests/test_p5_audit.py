from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from p5_audit_v2_cases import (
    CHANNELS,
    MODEL_PAIRS,
    TEST_EPISODES,
    aggregate_direct,
    make_case,
    refresh_producer_pin,
    refresh_rows,
    strict_json_write,
)

from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.experiments import p5_audit
from rl_attack.experiments.p5_audit import (
    ACCOUNTING_FIELDS,
    ATTACK_BUDGET_FIELDS,
    QUERY_BUDGET_FIELDS,
    REQUIRED_CELLS,
    ROW_FIELDS,
    InvalidP5Audit,
    load_p5_audit_config,
    run_p5_audit,
)


def test_v2_fixture_loads_and_runs(tmp_path: Path) -> None:
    config_path, _ = make_case(tmp_path)
    output = tmp_path / "audit"
    manifest = run_p5_audit(config_path, output_directory=output)
    assert manifest["status"] == "complete"
    assert manifest["formal_summary_eligible"] is False
    assert (output / "integration_results.json").is_file()


def test_formal_positive_matrix_and_detailed_metrics(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(
        tmp_path,
        formal_export=True,
        test_scope=False,
    )
    output = tmp_path / "formal"
    manifest = run_p5_audit(config_path, output_directory=output)
    summary = json.loads((output / "summaries.json").read_text(encoding="utf-8"))

    assert manifest["formal_summary_eligible"] is True
    assert manifest["robust_summary_eligible"] is True
    assert manifest["positive_claim_eligible"] is True
    assert summary["matrix"]["paired_complete"] is True
    assert summary["matrix"]["cell_count_per_hierarchy_unit"] == 13
    assert len(summary["paired_episode_worst_case"]) == (
        len(MODEL_PAIRS) * len(TEST_EPISODES)
    )
    first = summary["paired_episode_worst_case"][0]
    assert first["min_return"]["source_cells"][0]["attack"] == "STFA"
    assert first["max_collision_count"]["value"] == 3.0
    assert first["max_near_miss_count"]["value"] == 7.0
    assert first["max_safety_cost"]["value"] == 4.0
    assert summary["clean_cost"][
        "mean_return_cost_vs_frozen_anchor"
    ] == 1.0
    assert summary["detector"]["clean_false_positive_rate"] == 0.0
    assert summary["detector"]["curves"]["auroc"] == 1.0
    assert summary["detector"]["curves"]["auprc"] == 1.0
    assert summary["purifier"]["clean_action_agreement"] == 1.0
    cell = summary["cell_summaries"]["STFA/defense_aware"]
    assert cell["detector_true_positives"] == 80
    assert cell["detector_attack_opportunities"] == 80
    assert cell["purifier_repair_successes"] == 80
    assert cell["purifier_repair_opportunities"] == 80
    assert set(summary["latency"]["by_cell"]) == {
        f"{attack}/{adaptivity}"
        for attack, adaptivity in REQUIRED_CELLS
    }
    accounting = summary["accounting"]
    assert "total_queries" not in accounting
    assert "unified_budget" not in accounting
    assert set(accounting["totals_by_component"]) == set(ACCOUNTING_FIELDS)
    assert accounting["guard_episode_accounting_field_mapping"][
        "detector_policy_calls"
    ] == "detector_policy_queries"
    assert manifest["evidence_scope"][
        "public_driving_empirical_effectiveness"
    ] is False
    assert manifest["evidence_scope"]["sumo_empirical_effectiveness"] is False
    assert all(row["status"] == "complete" for row in rows)


def test_complete_formal_negative_result_is_not_silently_invalidated(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(
        tmp_path,
        formal_export=True,
        test_scope=False,
    )
    for row in rows:
        row["anchor_return"] = 110.0
    refresh_rows(config_path, rows)
    output = tmp_path / "negative"

    manifest = run_p5_audit(config_path, output_directory=output)
    summary = json.loads((output / "summaries.json").read_text(encoding="utf-8"))

    assert manifest["formal_summary_eligible"] is True
    assert manifest["robust_summary_eligible"] is True
    assert manifest["positive_claim_eligible"] is False
    assert summary["statistical_gate"]["h2"]["status"] == "failed"
    assert summary["statistical_gate"]["h3"]["status"] == "failed"
    assert manifest["eligibility"]["negative_results_are_summary_eligible"] is True
    assert not (output / "integration_results.json").exists()


def test_bootstrap_is_deterministic_and_ci_crossing_fails_h1(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path)
    for row in rows:
        if row["attack"] == "Clean":
            continue
        positive = row["episode_seed"] % 2 == 0
        row["detector_true_positives"] = 2
        row["detector_scores"] = [0.9, 0.9, 0.1, 0.1]
        row["detector_single_channel_true_positives"] = {
            channel: 4 if positive else 0 for channel in CHANNELS
        }
        row["purifier_repair_successes"] = 2
        row["purifier_repair_opportunities"] = 2
        row["no_purification_repair_successes"] = 0
        row["minimum_envelope_repair_successes"] = 1
    refresh_rows(config_path, rows)

    first = aggregate_direct(config_path, formal_eligible=True)
    second = aggregate_direct(config_path, formal_eligible=True)
    first_h1 = first["statistical_gate"]["h1"]

    assert first_h1 == second["statistical_gate"]["h1"]
    assert first_h1["status"] == "failed"
    assert any(
        comparison["confidence_interval"]["lower"] <= 0.0
        <= comparison["confidence_interval"]["upper"]
        for comparison in first_h1["comparisons"]
    )


def test_clean_cost_and_latency_fail_only_the_positive_claim(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path)
    for row in rows:
        row["anchor_return"] = 105.0
        row["latency_ms_by_component"]["end_to_end"] = [10.0] * 4
    refresh_rows(config_path, rows)

    summary = aggregate_direct(config_path, formal_eligible=True)

    assert summary["statistical_gate"]["h2"]["status"] == "failed"
    assert summary["statistical_gate"]["h3"]["status"] == "failed"
    assert (
        summary["statistical_gate"]["h3"]["constraints"][
            "worst_cell_end_to_end_p99_ms"
        ]
        == 10.0
    )


def test_minimum_units_and_unavailable_curves_are_pending(
    tmp_path: Path,
) -> None:
    seed_config, _ = make_case(
        tmp_path / "seed",
        formal_export=True,
        test_scope=False,
        model_pairs=((1, 11),),
    )
    seed_output = tmp_path / "seed" / "audit"
    seed_manifest = run_p5_audit(
        seed_config,
        output_directory=seed_output,
    )
    seed_result = json.loads(
        (seed_output / "integration_results.json").read_text(encoding="utf-8")
    )
    assert seed_manifest["formal_summary_eligible"] is False
    assert seed_manifest["positive_claim_eligible"] is False
    assert {
        seed_result["statistical_gate"][name]["status"]
        for name in ("h1", "h2", "h3")
    } == {"pending"}

    curve_config, rows = make_case(
        tmp_path / "curve",
        formal_export=True,
        test_scope=False,
    )
    for row in rows:
        row["detector_curve_status"] = "unavailable"
        row["detector_curve_unavailable_reason"] = (
            "producer did not export per-step detector scores"
        )
        row["detector_scores"] = None
        row["detector_labels"] = None
    refresh_rows(curve_config, rows)
    curve_output = tmp_path / "curve" / "audit"
    curve_manifest = run_p5_audit(
        curve_config,
        output_directory=curve_output,
    )
    curve_result = json.loads(
        (curve_output / "integration_results.json").read_text(
            encoding="utf-8"
        )
    )
    curves = curve_result["detector"]["curves"]
    assert curves["status"] == "unavailable"
    assert curves["auroc"] is None
    assert curves["auprc"] is None
    assert curve_result["statistical_gate"]["h1"]["status"] == "pending"
    assert curve_manifest["formal_summary_eligible"] is False


def test_csv_rows_have_cell_level_p50_p95_p99(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(
        tmp_path,
        formal_export=True,
        test_scope=False,
        row_format="csv",
    )
    output = tmp_path / "csv-audit"
    manifest = run_p5_audit(config_path, output_directory=output)
    summary = json.loads((output / "summaries.json").read_text(encoding="utf-8"))

    assert manifest["formal_summary_eligible"] is True
    expected = np.percentile(
        [
            sample
            for row in rows
            if row["attack"] == "Clean"
            for sample in row["latency_ms_by_component"]["end_to_end"]
        ],
        [50, 95, 99],
    )
    observed = summary["latency"]["by_cell"]["Clean/clean"]["end_to_end"]
    assert observed["p50_ms"] == pytest.approx(expected[0])
    assert observed["p95_ms"] == pytest.approx(expected[1])
    assert observed["p99_ms"] == pytest.approx(expected[2])


def test_csv_nested_json_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    config_path, _ = make_case(tmp_path, row_format="csv")
    loaded = load_p5_audit_config(config_path)
    with loaded.rows.path.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream, strict=True))
    records[0]["baseline_episode_metrics"] = (
        '{"Vanilla":{"episode_return":1,"episode_return":2}}'
    )
    with loaded.rows.path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(records)

    with pytest.raises(ValueError, match="duplicate JSON key 'episode_return'"):
        p5_audit._load_raw_rows(loaded.rows)


def test_detector_curve_negative_labels_bind_to_negative_opportunities(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path)
    clean = next(row for row in rows if row["attack"] == "Clean")
    clean["detector_scores"] = clean["detector_scores"][:-1]
    clean["detector_labels"] = clean["detector_labels"][:-1]
    refresh_rows(config_path, rows)

    with pytest.raises(
        InvalidP5Audit,
        match="detector negative labels/opportunities differ",
    ):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "negative-count-audit",
        )


def test_attacked_curve_cannot_hide_unaccounted_negative_samples(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path)
    attacked = next(row for row in rows if row["attack"] != "Clean")
    attacked["detector_scores"].append(0.1)
    attacked["detector_labels"].append(0)
    refresh_rows(config_path, rows)

    with pytest.raises(
        InvalidP5Audit,
        match="detector negative labels/opportunities differ",
    ):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "hidden-negative-audit",
        )


def test_zero_clean_purifier_calls_and_zero_distortion_are_valid(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path)
    for row in rows:
        if row["attack"] != "Clean":
            continue
        row["accounting"]["purifier_calls"] = 0
        row["purifier_l2_sum"] = 0.0
        row["purifier_linf_max"] = 0.0
    refresh_rows(config_path, rows)

    summary = aggregate_direct(config_path, formal_eligible=True)

    assert summary["purifier"]["clean_calls"] == 0
    assert summary["purifier"]["mean_clean_l2_distortion_per_call"] == 0.0
    assert summary["statistical_gate"]["h2"]["status"] == "passed"


def test_v1_config_rejected_and_evidence_scope_split(
    tmp_path: Path,
) -> None:
    config_path, _ = make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config["producer"]
    del config["model_seeds"]
    del config["statistics"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_p5_audit_config(config_path)

    config_path, _ = make_case(tmp_path / "scope")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evidence_scope"]["public_driving_empirical_effectiveness"] = True
    config["evidence_scope"]["public_driving_contract"] = False
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="public-driving empirical"):
        load_p5_audit_config(config_path)

    config_path, _ = make_case(tmp_path / "sumo")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evidence_scope"]["sumo_empirical_effectiveness"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids a SUMO"):
        load_p5_audit_config(config_path)


@pytest.mark.parametrize("group", ["episodes", "scenarios"])
def test_split_leakage_rejected_with_recomputed_hash(
    tmp_path: Path,
    group: str,
) -> None:
    config_path, _ = make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["splits"][group]["attacker_train"] = sorted(
        [
            config["splits"][group]["test"][0],
            *config["splits"][group]["attacker_train"][1:],
        ]
    )
    payload = {
        name: config["splits"][name]
        for name in ("episodes", "scenarios")
    }
    config["splits"]["contract_sha256"] = canonical_json_sha256(payload)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="cohort leakage"):
        load_p5_audit_config(config_path)


def test_producer_hash_rows_binding_source_and_scope_fail_closed(
    tmp_path: Path,
) -> None:
    config_path, _ = make_case(tmp_path / "hash")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    producer_path = (
        config_path.parent
        / config["producer"]["manifest"]["path"]
    )
    producer_path.write_text(
        producer_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="producer manifest SHA-256"):
        load_p5_audit_config(config_path)

    config_path, _ = make_case(tmp_path / "rows")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    producer_path = (
        config_path.parent
        / config["producer"]["manifest"]["path"]
    )
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    producer["rows"]["sha256"] = "0" * 64
    strict_json_write(producer_path, producer)
    refresh_producer_pin(config_path)
    with pytest.raises(ValueError, match="exact row export"):
        load_p5_audit_config(config_path)

    config_path, _ = make_case(tmp_path / "source")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    producer_path = (
        config_path.parent
        / config["producer"]["manifest"]["path"]
    )
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    producer["source"]["git_commit"] = "short"
    strict_json_write(producer_path, producer)
    refresh_producer_pin(config_path)
    with pytest.raises(ValueError, match="full 40-hex"):
        load_p5_audit_config(config_path)

    config_path, rows = make_case(tmp_path / "scope")
    for row in rows:
        row["test_scope"] = False
    refresh_rows(config_path, rows)
    with pytest.raises(InvalidP5Audit, match="producer test_scope"):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "scope" / "invalid",
        )


def test_defense_bundle_and_attacker_manifest_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    config_path, _ = make_case(tmp_path / "defense")
    loaded = load_p5_audit_config(config_path)
    defense_record = json.loads(
        loaded.defense.manifest.path.read_text(encoding="utf-8")
    )
    defense_record["bundle_manifest"]["claims"]["formal_robustness"] = True
    strict_json_write(loaded.defense.manifest.path, defense_record)
    tampered_defense = replace(
        loaded.defense,
        manifest=replace(
            loaded.defense.manifest,
            sha256=sha256_file(loaded.defense.manifest.path),
        ),
    )
    with pytest.raises(ValueError, match="claims overstate"):
        p5_audit._validate_rapid_guard_defense_manifest(
            tampered_defense,
            contracts=loaded.contracts,
            binding=loaded.defense_binding,
            splits=loaded.splits,
            model_seeds=loaded.model_seeds,
        )

    config_path, _ = make_case(tmp_path / "attacker")
    loaded = load_p5_audit_config(config_path)
    attack = "STFA"
    artifact = loaded.adaptive_attackers[attack]
    attacker_record = json.loads(
        artifact.manifest.path.read_text(encoding="utf-8")
    )
    attacker_record["training"]["test_episode_seeds_consumed"] = True
    strict_json_write(artifact.manifest.path, attacker_record)
    tampered_artifact = replace(
        artifact,
        manifest=replace(
            artifact.manifest,
            sha256=sha256_file(artifact.manifest.path),
        ),
    )
    artifacts = {
        **loaded.adaptive_attackers,
        attack: tampered_artifact,
    }
    with pytest.raises(ValueError, match="only the frozen attacker_train"):
        p5_audit._validate_adaptive_attacker_manifests(
            artifacts,
            bindings=loaded.adaptive_attack_bindings,
            defense=loaded.defense,
            defense_binding=loaded.defense_binding,
            budgets=loaded.attack_budgets,
            splits=loaded.splits,
            model_seeds=loaded.model_seeds,
        )


def test_missing_failed_cell_and_zero_opportunities_are_invalid_only(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path / "missing")
    refresh_rows(config_path, rows[:-1])
    output = tmp_path / "missing" / "audit"
    with pytest.raises(InvalidP5Audit, match="incomplete P5 matrix") as caught:
        run_p5_audit(config_path, output_directory=output)
    assert caught.value.manifest is not None
    assert {path.name for path in output.iterdir()} == {"manifest.json"}
    assert caught.value.manifest["formal_summary_eligible"] is False
    assert "summary" not in caught.value.manifest

    config_path, rows = make_case(tmp_path / "failed")
    rows[1]["status"] = "failed"
    refresh_rows(config_path, rows)
    output = tmp_path / "failed" / "audit"
    with pytest.raises(InvalidP5Audit, match="status must be complete"):
        run_p5_audit(config_path, output_directory=output)
    assert {path.name for path in output.iterdir()} == {"manifest.json"}

    config_path, rows = make_case(tmp_path / "opportunity")
    attacked = next(row for row in rows if row["attack"] != "Clean")
    attacked["detector_true_positives"] = 0
    attacked["detector_attack_opportunities"] = 0
    attacked["detector_single_channel_true_positives"] = {
        channel: 0 for channel in CHANNELS
    }
    attacked["detector_scores"] = [0.1, 0.1, 0.1, 0.1]
    attacked["detector_labels"] = [0, 0, 0, 0]
    attacked["purifier_repair_successes"] = 0
    attacked["purifier_repair_opportunities"] = 0
    attacked["no_purification_repair_successes"] = 0
    attacked["minimum_envelope_repair_successes"] = 0
    refresh_rows(config_path, rows)
    output = tmp_path / "opportunity" / "audit"
    with pytest.raises(InvalidP5Audit, match="no H1 opportunity"):
        run_p5_audit(config_path, output_directory=output)
    assert {path.name for path in output.iterdir()} == {"manifest.json"}


def test_schema_accounting_budget_and_latency_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path / "schema")
    rows[0]["unexpected"] = True
    refresh_rows(config_path, rows)
    with pytest.raises(InvalidP5Audit, match="schema mismatch"):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "schema" / "audit",
        )

    config_path, rows = make_case(tmp_path / "budget")
    attacked = next(row for row in rows if row["attack"] != "Clean")
    attacked["accounting"]["attacker_eot_samples"] = 21
    refresh_rows(config_path, rows)
    with pytest.raises(InvalidP5Audit, match="separately frozen budget"):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "budget" / "audit",
        )

    config_path, rows = make_case(tmp_path / "latency")
    rows[0]["latency_ms_by_component"]["detector"].pop()
    refresh_rows(config_path, rows)
    with pytest.raises(InvalidP5Audit, match="latency sample count"):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "latency" / "audit",
        )


def test_adaptivity_requires_corresponding_defense_query_evidence(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path / "nonadaptive")
    non_adaptive = next(
        row for row in rows if row["adaptivity"] == "non_adaptive"
    )
    non_adaptive["accounting"]["attacker_defense_forward_queries"] = 1
    refresh_rows(config_path, rows)
    with pytest.raises(InvalidP5Audit, match="non-adaptive attack queried"):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "nonadaptive" / "audit",
        )

    config_path, rows = make_case(tmp_path / "aware")
    aware = next(row for row in rows if row["adaptivity"] == "defense_aware")
    aware["accounting"]["attacker_defense_forward_queries"] = 0
    aware["accounting"]["attacker_defense_backward_queries"] = 0
    refresh_rows(config_path, rows)
    with pytest.raises(InvalidP5Audit, match="no defense query evidence"):
        run_p5_audit(
            config_path,
            output_directory=tmp_path / "aware" / "audit",
        )


def test_test_scope_and_injected_loader_are_permanently_ineligible(
    tmp_path: Path,
) -> None:
    config_path, rows = make_case(tmp_path / "rows")
    output = tmp_path / "rows" / "audit"
    manifest = run_p5_audit(config_path, output_directory=output)
    assert manifest["test_scope"] is True
    assert manifest["formal_summary_eligible"] is False
    assert manifest["positive_claim_eligible"] is False
    assert not (output / "summaries.json").exists()
    assert (output / "integration_results.json").is_file()

    config_path, rows = make_case(tmp_path / "injected")
    output = tmp_path / "injected" / "audit"
    manifest = run_p5_audit(
        config_path,
        output_directory=output,
        row_loader=lambda _config: rows,
    )
    assert manifest["dependency_injection"] == ["row_loader"]
    assert manifest["formal_summary_eligible"] is False
    assert manifest["positive_claim_eligible"] is False


def test_overwrite_atomic_publish_and_pinned_rows_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = make_case(tmp_path / "overwrite")
    occupied = tmp_path / "overwrite" / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        run_p5_audit(config_path, output_directory=occupied)

    config_path, _ = make_case(tmp_path / "atomic")
    output = tmp_path / "atomic" / "audit"
    original_replace = p5_audit.os.replace

    def fail_directory_publish(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        if Path(source).is_dir() and Path(destination) == output:
            raise OSError("injected directory publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(p5_audit.os, "replace", fail_directory_publish)
    with pytest.raises(OSError, match="publication failure"):
        run_p5_audit(config_path, output_directory=output)
    assert not output.exists()
    assert not list(output.parent.glob(".audit.stage-*"))
    monkeypatch.setattr(p5_audit.os, "replace", original_replace)

    config_path, _ = make_case(tmp_path / "hash")
    loaded = load_p5_audit_config(config_path)
    loaded.rows.path.write_text(
        loaded.rows.path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "hash" / "audit"
    with pytest.raises(InvalidP5Audit, match="rows SHA-256 mismatch"):
        run_p5_audit(loaded, output_directory=output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "invalid"
    assert {path.name for path in output.iterdir()} == {"manifest.json"}


def test_accounting_fields_keep_query_classes_non_fungible() -> None:
    assert set(QUERY_BUDGET_FIELDS) < set(ATTACK_BUDGET_FIELDS)
    assert set(ATTACK_BUDGET_FIELDS) < set(ACCOUNTING_FIELDS)
    assert {
        "detector_policy_calls",
        "proposal_calls",
        "semantic_projection_calls",
        "purification_attempts",
        "certificate_policy_calls",
        "ibp_bound_calls",
        "safety_critic_calls",
        "fallback_calls",
        "shield_calls",
        "defense_transform_calls",
    } <= set(ACCOUNTING_FIELDS)
