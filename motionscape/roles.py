"""Biological roles: the comparative framework of MOTIONSCAPE.

Core scientific question (fixed):

    用可探索的行为表型空间，研究不同拟蚁类群如何独立演化出
    "像蚂蚁一样运动"的能力。
    (How do ant-mimicking lineages independently evolve to move like ants?)

Every taxon in the atlas carries a **biological role** in the mimicry
comparison, kept strictly independent of its taxonomy:

    model                true ants — the mimicry models (Ant Behavioral
                         Reference Space)
    mimic                lineages with well-supported ant-mimicking
                         behavior, across INDEPENDENT evolutionary origins
    phylogenetic_control close non-mimic relatives — separates mimicry
                         from lineage-typical locomotion
    ecological_control   similar size/substrate/locomotion, non-mimicking —
                         separates mimicry from small-arthropod generic
                         movement
    unknown              not yet assessed

Roles are metadata for comparison and evaluation. They never enter the
encoders or the behavioral embedding (docs/SCIENTIFIC_ASSUMPTIONS.md):
first learn movement, then overlay biology — that is what makes
behavioral convergence observable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .schema import Episode

BIOLOGICAL_ROLES = ("model", "mimic", "phylogenetic_control",
                    "ecological_control", "unknown")

ROLE_NAMES = {"model": "Ant model", "mimic": "Ant mimic",
              "phylogenetic_control": "Non-mimic relative",
              "ecological_control": "Ecological control",
              "unknown": "Unassessed"}

ROLE_COLORS = {"model": "#e8a13c", "mimic": "#4fd1c5",
               "phylogenetic_control": "#b794f4",
               "ecological_control": "#a0aec0", "unknown": "#718096"}


@dataclass
class TaxonRole:
    """Role assignment for one taxon, with its evidence trail (§41)."""
    taxon: str                            # canonical species or clade name
    role: str                             # one of BIOLOGICAL_ROLES
    mimicry_system_id: str | None = None  # e.g. "siler_southeast_asia"
    putative_model_taxon: str | None = None
    mimicry_evidence: str | None = None   # published citation / description
    evidence_kind: str | None = None      # published | morphological | behavioral | ecological
    confidence: float = 0.0
    reference_source: str | None = None   # DOI / URL / dataset the judgment rests on
    lineage_note: str | None = None       # independent-origin grouping note

    def to_dict(self) -> dict:
        return asdict(self)


# Seed registry for the taxa currently in the atlas. Sources: Shamble et
# al. 2017 (Dryad fd612) species metadata; Zeng et al. 2023 (iScience)
# species metadata. Extending this registry is a data-curation act with a
# citation, never a developer hunch (§41).
DEFAULT_TAXON_ROLES: list[TaxonRole] = [
    TaxonRole(
        taxon="ant", role="model",
        mimicry_evidence="generic ant entry: any ant episode not yet assigned "
                         "to a species-level model",
        reference_source="Shamble et al. 2017 Dryad fd612; Zeng et al. 2023"),
    TaxonRole(
        taxon="Lasius", role="model",
        putative_model_taxon="Lasius (formicine ant)",
        reference_source="Shamble et al. 2017 Dryad fd612"),
    # the sympatric ant community of the Siler showcase (Zeng et al. 2023)
    TaxonRole(taxon="Crematogaster egidyi", role="model",
              mimicry_system_id="siler_southeast_asia",
              reference_source="10.1016/j.isci.2023.106653"),
    TaxonRole(taxon="Meranoplus bicolor", role="model",
              mimicry_system_id="siler_southeast_asia",
              reference_source="10.1016/j.isci.2023.106653"),
    TaxonRole(taxon="Technomyrmex sp.", role="model",
              mimicry_system_id="siler_southeast_asia",
              reference_source="10.1016/j.isci.2023.106653"),
    TaxonRole(taxon="Polyrhachis jianghuaensis", role="model",
              mimicry_system_id="siler_southeast_asia",
              reference_source="10.1016/j.isci.2023.106653"),
    TaxonRole(taxon="Polyrhachis dives", role="model",
              mimicry_system_id="siler_southeast_asia",
              reference_source="10.1016/j.isci.2023.106653"),
    TaxonRole(
        taxon="Siler collingwoodi", role="mimic",
        mimicry_system_id="siler_southeast_asia",
        putative_model_taxon="sympatric ants (Polyrhachis, Crematogaster; Zeng et al. 2023)",
        mimicry_evidence="Zeng et al. 2023 iScience: locomotor mimicry in "
                         "Siler collingwoodi",
        evidence_kind="published", confidence=0.9,
        reference_source="10.1016/j.isci.2023.106653",
        lineage_note="salticid lineage; independent origin #1 showcase"),
    TaxonRole(
        taxon="Myrmarachne formicaria", role="mimic",
        mimicry_system_id="myrmarachnini",
        putative_model_taxon="sympatric ants (formicines et al.)",
        mimicry_evidence="recognized ant-mimicking salticid; behavioral "
                         "mimicry documented by Shamble et al. 2017",
        evidence_kind="published", confidence=0.9,
        reference_source="10.5061/dryad.fd612; Shamble et al. 2017 Current Biology",
        lineage_note="Myrmarachnini lineage; independent origin (Phase 2 anchor)"),
    TaxonRole(
        taxon="Phintelloides versicolor", role="phylogenetic_control",
        mimicry_evidence="non-mimicking salticid used as close-relative "
                         "control in Zeng et al. 2023",
        evidence_kind="published", confidence=0.9,
        reference_source="10.1016/j.isci.2023.106653"),
    TaxonRole(
        taxon="Salticus senicus", role="phylogenetic_control",
        mimicry_evidence="non-mimicking salticid control (Shamble et al. 2017)",
        evidence_kind="published", confidence=0.9,
        reference_source="10.5061/dryad.fd612"),
]


class RoleRegistry:
    """Taxon -> role assignments. Stored as JSON so curation is appendable
    and citable; roles are metadata and never touch the embedding."""

    def __init__(self, roles: list[TaxonRole] | None = None):
        self.roles: dict[str, TaxonRole] = {}
        for r in (roles if roles is not None else DEFAULT_TAXON_ROLES):
            self.roles[r.taxon.lower()] = r

    @classmethod
    def load(cls, path: str | Path) -> "RoleRegistry":
        """Load a curated registry; a missing file yields the default
        seed registry (never an empty one — roles must stay resolvable)."""
        p = Path(path)
        if not p.exists():
            return cls()
        reg = cls(roles=[])
        data = json.loads(p.read_text(encoding="utf-8"))
        for d in data.get("roles", []):
            r = TaxonRole(**d)
            reg.roles[r.taxon.lower()] = r
        return reg

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "roles": [r.to_dict() for r in self.roles.values()],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def role_for(self, taxon: str | None) -> str:
        """Resolve a role for an arbitrary taxon string: exact species
        match first, then prefix (genus), then the generic 'ant' entry."""
        if not taxon:
            return "unknown"
        key = taxon.strip().lower()
        if key in self.roles:
            return self.roles[key].role
        for name, r in self.roles.items():
            if key.startswith(name) or name.startswith(key):
                return r.role
        if "ant" in key:
            return "model"
        return "unknown"

    def details_for(self, taxon: str | None) -> TaxonRole | None:
        if not taxon:
            return None
        key = taxon.strip().lower()
        if key in self.roles:
            return self.roles[key]
        for name, r in self.roles.items():
            if key.startswith(name) or name.startswith(key):
                return r
        return None

    def apply_to_episodes(self, episodes: list[Episode]) -> int:
        """Denormalize roles onto episodes (biological_role field): try the
        species detail first, then the coarse label. Returns the number of
        episodes whose role was resolved."""
        n = 0
        for e in episodes:
            taxon = (e.bio_label_detail or {}).get("species")
            role = self.role_for(taxon)
            if role == "unknown":
                role = self.role_for(e.bio_label)
            e.biological_role = role
            if role != "unknown":
                n += 1
        return n


def default_registry_path(root: str | Path) -> Path:
    return Path(root) / "roles.json"
