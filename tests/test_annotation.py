"""Annotation system: persistence, machine/human separation, QC states."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinemimic.annotation import (AnnotationRecord, append_record, apply_log,
                                    apply_record, read_log, summary)
from kinemimic.classify import classify_episode, label_episodes
from kinemimic.schema import BIO_LABELS, QC_STATES, Episode
from kinemimic.synth import generate_paths


def mk_ep(label="unknown"):
    p = generate_paths(300, 960, 540, "ant", n_animals=1, seed=7)[0]
    return Episode(episode_id="ep_test_1", frames=list(range(len(p))),
                   centroids_px=p.tolist(), start_frame=0, end_frame=len(p),
                   fps=30.0, bbox_sizes_px=[[27.0, 9.0]] * len(p),
                   detection_confidence=[0.9] * len(p), bio_label=label)


def test_record_persistence_append_only(tmp_path):
    root = tmp_path / "store"
    append_record(root, AnnotationRecord(episode_id="ep_test_1", human_label="ant",
                                         qc_state="accepted", annotator="alice"))
    append_record(root, AnnotationRecord(episode_id="ep_test_1", human_label="siler",
                                         qc_state="accepted", annotator="bob"))
    lines = read_log(root)
    assert len(lines) == 2                      # both kept: append-only history
    assert [r.annotator for r in lines] == ["alice", "bob"]
    log = (root / "annotations" / "annotations.jsonl").read_text(encoding="utf-8")
    assert len(log.strip().splitlines()) == 2   # nothing overwritten


def test_latest_record_wins(tmp_path):
    root = tmp_path / "store"
    ep = mk_ep()
    append_record(root, AnnotationRecord(episode_id="ep_test_1", human_label="ant",
                                         annotator="alice"))
    append_record(root, AnnotationRecord(episode_id="ep_test_1", human_label="siler",
                                         annotator="bob", qc_state="ambiguous_taxon"))
    apply_log([ep], root)
    assert ep.human_label == "siler"            # latest wins
    assert ep.annotator == "bob"
    assert ep.annotation_status == "ambiguous_taxon"


def test_machine_vs_human_separation(tmp_path):
    root = tmp_path / "store"
    ep = mk_ep()
    label_episodes([ep])                        # machine pre-classification
    machine = (ep.machine_label, ep.machine_confidence, ep.machine_source)
    assert machine[0] in BIO_LABELS and machine[2].startswith("model:")
    # human disagrees with the machine
    append_record(root, AnnotationRecord(episode_id="ep_test_1",
                                         human_label="other_spider", annotator="alice"))
    apply_log([ep], root)
    # machine prediction untouched, human annotation recorded, effective flips
    assert (ep.machine_label, ep.machine_confidence, ep.machine_source) == machine
    assert ep.human_label == "other_spider"
    assert ep.effective_label() == "other_spider"
    assert ep.annotation_status == "accepted"


def test_machine_labeling_never_overwrites_human(tmp_path):
    ep = mk_ep()
    ep.human_label = "ant"
    ep.annotator = "alice"
    label_episodes([ep])
    assert ep.machine_label in BIO_LABELS       # prediction still computed...
    assert ep.effective_label() == "ant"        # ...but human wins


def test_qc_state_transitions(tmp_path):
    root = tmp_path / "store"
    ep = mk_ep()
    seen = []
    for qc in ("rejected", "tracking_failure", "severe_occlusion", "edge_effect",
               "too_short", "ambiguous_taxon", "accepted", "unreviewed"):
        append_record(root, AnnotationRecord(episode_id="ep_test_1", qc_state=qc))
        apply_log([ep], root)
        assert ep.annotation_status == qc
        seen.append(qc)
    assert set(seen) <= set(QC_STATES)
    # rejected episodes keep their observation data (never deleted)
    assert len(ep.centroids_px) == 300


def test_invalid_label_and_state_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        append_record(tmp_path, AnnotationRecord(episode_id="e", human_label="bat"))
    with pytest.raises(ValueError):
        append_record(tmp_path, AnnotationRecord(episode_id="e", qc_state="kind_of_ok"))


def test_summary_counts(tmp_path):
    root = tmp_path / "store"
    eps = [mk_ep(), mk_ep(), mk_ep()]
    eps[0].machine_label = "ant"
    append_record(root, AnnotationRecord(episode_id=eps[1].episode_id,
                                         human_label="siler"))
    apply_log(eps, root)
    s = summary(eps)
    assert s["total"] == 3 and s["reviewed"] == 1
    assert s["by_effective_label"] == {"ant": 1, "siler": 1, "unknown": 1}
    assert s["n_human_labeled"] == 1 and s["n_machine_only"] == 1


def test_v01_legacy_migration():
    d = {"bio_label": "ant", "bio_label_confidence": 1.0,
         "bio_label_source": "human:dataset_metadata"}
    ep = Episode.from_dict(d)
    assert ep.human_label == "ant" and ep.annotator == "human:dataset_metadata"
    assert ep.annotation_status == "accepted"
    d2 = {"bio_label": "siler", "bio_label_source": "model:heuristic-v1"}
    ep2 = Episode.from_dict(d2)
    assert ep2.machine_label == "siler" and ep2.human_label is None


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(tmp_path := Path(f"/tmp/ms_test_{name}"))
            print(f"{name} OK")
