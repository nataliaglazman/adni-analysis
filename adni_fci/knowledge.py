"""Background knowledge and the restricted conditional-independence test.

Background knowledge forbids edge directions that are biologically implausible:

1. Demographics (and technical covariates) are exogenous: nothing causes them.
2. Cognition is downstream: cognitive scores cannot cause biomarkers or brain volumes.
3. Optional rules, switched on per setting (see ``Setting``): no edges among cognitive
   scores, MRI volumes cannot cause fluid biomarkers, and NfL/GFAP cannot cause
   amyloid/tau markers.

Forbidding *both* directions of a pair (e.g. two demographics, or two cognitive scores
with ``forbid_cognition_to_cognition``) removes that adjacency altogether.
"""
from __future__ import annotations

from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import CIT, CIT_Base
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from .settings import Setting

NEURODEGENERATION_MARKERS = ("NfL_Q", "GFAP_Q", "NfL", "GFAP")
AMYLOID_TAU_MARKERS = ("AB42_AB40_ratio", "pT217_F", "ABETA42_ABETA40_ratio", "PTAU", "META_TEMPORAL_SUVR")


def forbidden_edges(columns, setting: Setting) -> set[tuple[str, str]]:
    """All (cause, effect) pairs ruled out for these columns, by variable name."""
    present = set(columns)
    forbidden: set[tuple[str, str]] = set()

    def forbid(causes, effects):
        forbidden.update((c, e) for c in causes for e in effects
                         if c in present and e in present and c != e)

    forbid(columns, setting.roots)
    if setting.cognition_is_sink:
        forbid(setting.cognitive, [*setting.biomarkers, *setting.mri])
    if setting.forbid_cognition_to_cognition:
        forbid(setting.cognitive, setting.cognitive)
    if setting.forbid_mri_to_biomarkers:
        forbid(setting.mri, setting.biomarkers)
    if setting.amyloid_tau_cascade:
        forbid(NEURODEGENERATION_MARKERS, AMYLOID_TAU_MARKERS)
    return forbidden


def build_background_knowledge(columns, setting: Setting) -> BackgroundKnowledge:
    """causal-learn background knowledge for data whose i-th column is ``columns[i]``.

    causal-learn matches nodes by name, and ``fci()`` names its nodes X1..Xp.
    """
    columns = list(columns)
    nodes = [GraphNode(f"X{i + 1}") for i in range(len(columns))]
    index = {c: i for i, c in enumerate(columns)}
    bk = BackgroundKnowledge()
    for cause, effect in sorted(forbidden_edges(columns, setting)):
        bk.add_forbidden_by_node(nodes[index[cause]], nodes[index[effect]])
    return bk


class RestrictedCIT(CIT_Base):
    """CI test that never conditions on the variables at ``forbidden`` column indices.

    FCI calls ``test(X, Y, S)``; this wrapper drops forbidden indices from S before
    delegating. Keeping cognitive scores out of conditioning sets avoids spurious
    independencies from conditioning on downstream colliders such as ADAS-Cog.
    """

    def __init__(self, base: CIT_Base, forbidden):
        super().__init__(base.data)
        self.base = base
        self.forbidden = frozenset(int(i) for i in forbidden)
        self.method = getattr(base, "method", None)

    def __call__(self, X, Y, condition_set=None):
        if condition_set is not None:
            condition_set = [c for c in condition_set if int(c) not in self.forbidden]
        return self.base(X, Y, condition_set)


def make_ci_test(data, columns, setting: Setting) -> CIT_Base:
    """The setting's CI test on ``data``, restricted to exclude cognitive conditioning if requested."""
    base = CIT(data, setting.ci_test, **setting.ci_test_kwargs)
    cognitive = [i for i, c in enumerate(columns) if c in setting.cognitive]
    if setting.restrict_conditioning_on_cognition and cognitive:
        return RestrictedCIT(base, cognitive)
    return base
