"""The sweep script parses `gevbench eval` reports; the sample is real output."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "lora_scale_sweep.py"
spec = importlib.util.spec_from_file_location("lora_scale_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

REPORT = """
=== aegis2  (250 items) ===
=== gev / google/gemma-4-E4B-it ===
wall time          29.4s  (8.52 items/s)
calls              250
transient retries  0

-- decision
   accuracy 0.812   log-loss 0.5224   brier 0.3087   ECE 0.1132
   conf>=0.5: accuracy 0.812 on 100% of items
   reliability (top-probability bin -> accuracy, n):
     [0.5,0.6)  conf 0.552  acc 0.286  n=7  gap +0.266


=== massive-de-DE  (1,350 items) ===
=== gev / google/gemma-4-E4B-it ===
wall time          55.7s  (6.28 items/s)

-- decision
   accuracy 0.851   log-loss 0.5344   brier 0.2289   ECE 0.0756
"""


def test_each_subset_gets_its_own_accuracy_not_a_neighbours():
    results = sweep.parse_eval(REPORT)
    assert results == {
        "aegis2": {"accuracy": 0.812, "ece": 0.1132, "items": 250},
        "massive-de-DE": {"accuracy": 0.851, "ece": 0.0756, "items": 1350},
    }


def test_the_table_has_one_column_per_scale_and_a_macro_over_complete_rows():
    by_scale = {
        0.0: {"a": {"accuracy": 0.8}, "b": {"accuracy": 0.2}},
        1.0: {"a": {"accuracy": 0.6}, "b": {"accuracy": 0.4}, "c": {"accuracy": 0.9}},
    }
    lines = sweep.table([0.0, 1.0], by_scale).splitlines()
    assert lines[0].split() == ["subset", "0", "1"]
    assert lines[-1].split() == ["macro", "(2", "subsets)", "0.500", "0.500"]
    assert [row.split()[0] for row in lines[2:5]] == ["a", "b", "c"]
    assert lines[4].split()[1:] == ["-", "0.900"], "a subset missing at a scale shows a dash"
