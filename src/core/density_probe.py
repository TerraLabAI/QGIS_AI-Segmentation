




















from __future__ import annotations

import math
from dataclasses import dataclass, field

BRANCH_STAY = "stay"
BRANCH_COARSEN = "coarsen"
BRANCH_REFINE = "refine"



SKIP_OFF = "off"
SKIP_FAMILY = "family"
SKIP_EXEMPLAR = "exemplar"
SKIP_USER_STEP = "user_step"
SKIP_HEADLESS = "headless"
SKIP_ZONE_SMALL = "zone_small"
SKIP_ZONE_LARGE = "zone_large"
SKIP_REPLANNED = "replanned"
STAY_FEW_ANSWERS = "few_answers"
STAY_GATES = "gates"
STAY_SMALL_CHANGE = "small_change"
STAY_REFINE_TILES = "refine_tiles"


@dataclass(frozen=True)
class DensityFamily:





    name: str
    words: frozenset
    coarsen_max_ground_m: float = 0.0
    refine_ground_m: float = 0.0


@dataclass(frozen=True)
class DensityProbeConfig:


    probe_tiles: int
    probe_tiles_min: int
    min_zone_tiles: int
    max_zone_tiles: int
    target_masks: float
    law_exponent: float
    coarsen_median_min: float
    coarsen_median_max: float
    coarsen_p85_max: float
    veto_tile_masks: float
    refine_p85_min: float
    refine_median_min: float
    refine_max_tiles: int
    refine_max_tiles_free: int
    min_change_ratio: float
    families: tuple = field(default_factory=tuple)
    exclude_words: frozenset = frozenset()


@dataclass(frozen=True)
class DensityDecision:



    branch: str
    side_m: float
    reason: str
    k: int = 0
    median: float = 0.0
    p85: float = 0.0
    max: float = 0.0

    def props(self) -> dict:

        return {
            "density_branch": self.branch,
            "density_reason": self.reason,
            "density_k": int(self.k),
            "density_median": round(float(self.median), 1),
            "density_p85": round(float(self.p85), 1),
            "density_max": round(float(self.max), 1),
            "density_side_m": int(round(self.side_m)),
        }


def _num(value, lo: float, hi: float):


    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if not math.isfinite(val) or val < lo or val > hi:
        return None
    return val


def _words(value) -> frozenset:
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(
        w.strip().lower() for w in value if isinstance(w, str) and w.strip())


def config_from_block(block) -> DensityProbeConfig | None:



    if not isinstance(block, dict) or block.get("enabled") is not True:
        return None
    k = _num(block.get("probe_tiles"), 1, 64)
    k_min = _num(block.get("probe_tiles_min"), 1, 64)
    zmin = _num(block.get("min_zone_tiles"), 1, 100000)
    zmax = _num(block.get("max_zone_tiles"), 1, 100000)
    target = _num(block.get("target_masks"), 1, 1000)
    expo = _num(block.get("law_exponent"), 0.05, 2.0)
    c_lo = _num(block.get("coarsen_median_min"), 0, 1000)
    c_hi = _num(block.get("coarsen_median_max"), 0, 1000)
    c_p85 = _num(block.get("coarsen_p85_max"), 0, 1000)
    veto = _num(block.get("veto_tile_masks"), 1, 1000)
    need = (k, k_min, zmin, zmax, target, expo, c_lo, c_hi, c_p85, veto)
    if any(v is None for v in need):
        return None
    if k_min > k or zmin > zmax or c_lo > c_hi:
        return None

    r_p85 = _num(block.get("refine_p85_min", 0), 0, 1000) or 0.0
    r_med = _num(block.get("refine_median_min", 0), 0, 1000) or 0.0
    r_tiles = _num(block.get("refine_max_tiles", 0), 0, 1000000) or 0.0
    r_free = _num(block.get("refine_max_tiles_free", 0), 0, 1000000) or 0.0
    change = _num(block.get("min_change_ratio", 0.1), 0.0, 1.0)
    if change is None:
        change = 0.1
    families = []
    fams = block.get("families")
    if isinstance(fams, dict):
        for name, spec in fams.items():
            if not isinstance(spec, dict) or not isinstance(name, str):
                continue
            words = _words(spec.get("words"))
            if not words:
                continue
            families.append(DensityFamily(
                name=name,
                words=words,
                coarsen_max_ground_m=_num(
                    spec.get("max_ground_m", 0), 0, 5000) or 0.0,
                refine_ground_m=_num(
                    spec.get("refine_ground_m", 0), 0, 5000) or 0.0,
            ))
    if not families:
        return None
    return DensityProbeConfig(
        probe_tiles=int(k), probe_tiles_min=int(k_min),
        min_zone_tiles=int(zmin), max_zone_tiles=int(zmax),
        target_masks=target, law_exponent=expo,
        coarsen_median_min=c_lo, coarsen_median_max=c_hi,
        coarsen_p85_max=c_p85, veto_tile_masks=veto,
        refine_p85_min=r_p85, refine_median_min=r_med,
        refine_max_tiles=int(r_tiles), refine_max_tiles_free=int(r_free),
        min_change_ratio=change,
        families=tuple(families),
        exclude_words=_words(block.get("exclude_words")),
    )


def family_for(word: str, config: DensityProbeConfig) -> DensityFamily | None:



    text = " ".join((word or "").lower().replace("_", " ").split())
    if not text:
        return None
    tokens = set(text.split())
    for bad in config.exclude_words:
        if bad in tokens or (" " in bad and bad in text):
            return None
    for fam in config.families:
        if text in fam.words:
            return fam
    return None


def probe_k(n_tiles: int, config: DensityProbeConfig) -> int:

    return max(0, min(config.probe_tiles,
                      max(config.probe_tiles_min, n_tiles // 2), n_tiles))


def probe_order(centres: list, k: int) -> list[int]:



    n = len(centres)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    gx = 4
    gy = max(1, k // gx)
    xs = [c[0] for c in centres]
    ys = [c[1] for c in centres]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    picked: list[int] = []
    taken = set()
    for i in range(gx):
        for j in range(gy):
            if len(picked) >= k:
                break
            cx = x0 + (i + 0.5) * (x1 - x0) / gx
            cy = y0 + (j + 0.5) * (y1 - y0) / gy
            best = None
            best_d = None
            for idx, (x, y) in enumerate(centres):
                if idx in taken:
                    continue
                d = (x - cx) ** 2 + (y - cy) ** 2
                if best_d is None or d < best_d:
                    best, best_d = idx, d
            if best is not None:
                picked.append(best)
                taken.add(best)
    return picked


def probe_stats(counts: list) -> tuple[float, float, float]:


    vals = sorted(float(c) for c in counts)
    if not vals:
        return 0.0, 0.0, 0.0
    n = len(vals)
    mid = n // 2
    median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    p85 = vals[-2] if n >= 2 else vals[-1]
    return median, p85, vals[-1]


def decide(counts: list, side_m: float, family: DensityFamily,
           config: DensityProbeConfig) -> DensityDecision:


    k = len(counts)
    median, p85, top = probe_stats(counts)
    base = {"k": k, "median": median, "p85": p85, "max": top}
    if k < config.probe_tiles_min or side_m <= 0:
        return DensityDecision(BRANCH_STAY, side_m, STAY_FEW_ANSWERS, **base)


    if (family.refine_ground_m > 0 and config.refine_p85_min > 0
            and p85 >= config.refine_p85_min
            and median >= config.refine_median_min):
        want = family.refine_ground_m
        if want < side_m * (1.0 - config.min_change_ratio):
            return DensityDecision(BRANCH_REFINE, want, BRANCH_REFINE, **base)
        return DensityDecision(BRANCH_STAY, side_m, STAY_SMALL_CHANGE, **base)
    if family.coarsen_max_ground_m > 0:
        passes = (config.coarsen_median_min <= median <= config.coarsen_median_max
                  and p85 <= config.coarsen_p85_max
                  and top < config.veto_tile_masks)
        if passes and median > 0:
            want = side_m * (config.target_masks / median) ** config.law_exponent
            want = max(side_m, min(family.coarsen_max_ground_m, want))
            if want > side_m * (1.0 + config.min_change_ratio):
                return DensityDecision(BRANCH_COARSEN, want, BRANCH_COARSEN, **base)
            return DensityDecision(BRANCH_STAY, side_m, STAY_SMALL_CHANGE, **base)
    return DensityDecision(BRANCH_STAY, side_m, STAY_GATES, **base)
