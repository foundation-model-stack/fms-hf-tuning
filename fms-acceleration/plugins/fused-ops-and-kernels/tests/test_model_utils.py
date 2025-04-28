# Third Party
from fms_acceleration.model_patcher import ModelPatcherRule

# First Party
from fms_acceleration_foak.utils import filter_mp_rules


def test_filter_mp_rules():

    rules = [
        ModelPatcherRule(rule_id="rule1_a", forward=lambda x: x),
        ModelPatcherRule(rule_id="rule2_a", forward=lambda x: x),
        ModelPatcherRule(rule_id="rule1_b", forward=lambda x: x),
    ]

    # - we will keep only a's
    assert len(filter_mp_rules(rules, filter_endswith={"a"}, drop=False)) == 2

    # - we will drop only a's
    assert len(filter_mp_rules(rules, filter_endswith={"a"}, drop=True)) == 1

    # - we will drop only b's
    assert len(filter_mp_rules(rules, filter_endswith={"b"}, drop=True)) == 2

    # - we will keep a and b's
    assert len(filter_mp_rules(rules, filter_endswith={"a", "b"}, drop=False)) == 3

    # - we will drop both a and b's
    assert len(filter_mp_rules(rules, filter_endswith={"a", "b"}, drop=True)) == 0
