"""Biological roles: registry resolution, episode application, atlas meta.

Roles are the comparative framework (model / mimic / phylogenetic
control / ecological control) — independent of taxonomy, never used by
encoders, always carrying their evidence trail.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinemimic.roles import (BIOLOGICAL_ROLES, RoleRegistry, TaxonRole,
                               default_registry_path)
from kinemimic.schema import Episode


def _ep(species=None, label="unknown"):
    e = Episode(episode_id=f"ep_{species or label}",
                bio_label=label,
                bio_label_detail=({"species": species} if species else {}))
    return e


def test_default_registry_resolves_known_taxa():
    reg = RoleRegistry()
    assert reg.role_for("Siler collingwoodi") == "mimic"
    assert reg.role_for("Myrmarachne formicaria") == "mimic"
    assert reg.role_for("Lasius") == "model"
    assert reg.role_for("Crematogaster egidyi") == "model"     # Siler's models
    assert reg.role_for("Polyrhachis dives") == "model"
    assert reg.role_for("Phintelloides versicolor") == "phylogenetic_control"
    assert reg.role_for("Salticus senicus") == "phylogenetic_control"


def test_unknown_taxon_stays_unknown():
    reg = RoleRegistry()
    assert reg.role_for("some random beetle") == "unknown"
    assert reg.role_for(None) == "unknown"


def test_role_independent_of_taxonomy_label():
    """A label like 'ant' may carry the model role, but roles are stored as
    their own metadata field, never as a taxonomy rewrite."""
    reg = RoleRegistry()
    e = _ep(species="Crematogaster egidyi", label="ant")
    reg.apply_to_episodes([e])
    assert e.biological_role == "model"
    assert e.bio_label == "ant"                      # taxonomy untouched
    assert e.biological_role in BIOLOGICAL_ROLES


def test_apply_prefers_species_detail_then_label():
    reg = RoleRegistry()
    by_species = _ep(species="Siler collingwoodi", label="unknown")
    by_label = _ep(species=None, label="ant")
    reg.apply_to_episodes([by_species, by_label])
    assert by_species.biological_role == "mimic"     # resolved via species
    assert by_label.biological_role == "model"       # resolved via label


def test_registry_persistence_roundtrip(tmp_path):
    reg = RoleRegistry()
    reg.roles["myrmarachne_sp_b"] = TaxonRole(
        taxon="Myrmarachne sp. B", role="mimic",
        mimicry_system_id="myrmarachnini",
        putative_model_taxon="unknown",
        mimicry_evidence="candidate mimic — awaiting citation",
        evidence_kind="behavioral", confidence=0.4,
        reference_source="field observation 2026")
    path = reg.save(tmp_path / "roles.json")
    assert path.exists()
    reg2 = RoleRegistry.load(path)
    assert reg2.role_for("Myrmarachne sp. B") == "mimic"
    d = reg2.details_for("Myrmarachne sp. B")
    assert d.mimicry_system_id == "myrmarachnini"
    assert d.evidence_kind == "behavioral"
    # every stored role is a valid role name
    assert all(r.role in BIOLOGICAL_ROLES for r in reg2.roles.values())


def test_load_missing_file_yields_defaults(tmp_path):
    reg = RoleRegistry.load(tmp_path / "does_not_exist.json")
    assert reg.role_for("Siler collingwoodi") == "mimic"


def test_importers_assign_roles():
    """Real importers: gold labels resolve to model / mimic / control."""
    from kinemimic.shamble import load_shamble_episodes
    from kinemimic.zeng import load_zeng_episodes
    sh = load_shamble_episodes(
        "data/external/shamble2017/OverallMovement/OverallMovement/data/"
        "data_folders_5_to_18_v3.mat")
    roles = {e.biological_role for e in sh}
    assert roles == {"model", "mimic", "phylogenetic_control"}
    z = load_zeng_episodes(
        "data/external/Zeng_et_al_2023_Gait analysis_raw data.xlsx")
    assert {e.biological_role for e in z} == {"model", "mimic",
                                              "phylogenetic_control"}
    # the five sympatric ants of the Siler system are models
    assert all(e.biological_role == "model" for e in z
               if e.bio_label == "ant")


def test_atlas_embeds_roles_for_reveal(tmp_path):
    """The atlas meta carries taxon->role mapping and per-episode roles so
    the UI can Reveal roles, not only species."""
    from kinemimic.atlas import build_atlas
    from kinemimic.benchmark import synthetic_episodes
    from kinemimic.roles import RoleRegistry
    eps = synthetic_episodes(24, seed=5)
    for e in eps:
        e.bio_label = "ant"                      # exercise the role resolver
    build_atlas(eps, tmp_path / "atlas", make_meta_files=False)
    data = json.loads((tmp_path / "atlas" / "data.json").read_text(encoding="utf-8"))
    assert "roles" in data["meta"]
    assert data["meta"]["roles"]["ant"]["role"] == "model"
    assert (tmp_path / "atlas" / "roles.json").exists()
    assert all(e["r"] in BIOLOGICAL_ROLES for e in data["episodes"])


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(Path(f"/tmp/ms_test_{name}"))
            print(f"{name} OK")
