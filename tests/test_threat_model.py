import pytest

from rl_attack.core.threat_model import (
    AttackKnowledge,
    AttackObjective,
    AttackStage,
    ThreatModel,
)


def test_threat_model_validates_temporal_fraction():
    threat = ThreatModel(
        stage=AttackStage.OBSERVATION,
        knowledge=AttackKnowledge.WHITE_BOX,
        objective=AttackObjective.POLICY_DIVERGENCE,
        temporal_attack_fraction=0.25,
    )
    assert threat.temporal_attack_fraction == 0.25

    with pytest.raises(ValueError):
        ThreatModel(
            stage=AttackStage.OBSERVATION,
            knowledge=AttackKnowledge.WHITE_BOX,
            objective=AttackObjective.POLICY_DIVERGENCE,
            temporal_attack_fraction=1.1,
        )

