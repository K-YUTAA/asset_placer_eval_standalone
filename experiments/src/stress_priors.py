from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Vec2 = Tuple[float, float]


def clipped_gaussian(rng: random.Random, sigma: float, clip_abs: float) -> float:
    sigma = float(sigma)
    clip_abs = abs(float(clip_abs))
    if sigma <= 0.0 or clip_abs <= 0.0:
        return 0.0
    last = 0.0
    for _ in range(8):
        last = rng.gauss(0.0, sigma)
        if abs(last) <= clip_abs:
            return last
    return max(-clip_abs, min(clip_abs, last))


def choose_count_from_distribution(
    rng: random.Random,
    distribution: Sequence[Dict[str, Any]],
    *,
    count_key: str,
    default: int,
) -> int:
    if not isinstance(distribution, Sequence) or not distribution:
        return int(default)
    weights: List[float] = []
    counts: List[int] = []
    for item in distribution:
        if not isinstance(item, dict):
            continue
        try:
            count = int(item.get(count_key, default))
        except Exception:
            count = int(default)
        weight = float(item.get("probability", 0.0))
        if weight <= 0.0:
            continue
        counts.append(count)
        weights.append(weight)
    if not counts:
        return int(default)
    total = sum(weights)
    if total <= 0.0:
        return counts[0]
    pick = rng.random() * total
    acc = 0.0
    for count, weight in zip(counts, weights):
        acc += weight
        if pick <= acc:
            return count
    return counts[-1]


def choose_weighted_item(
    rng: random.Random,
    items: Sequence[Dict[str, Any]],
    *,
    weight_key: str = "weight",
) -> Optional[Dict[str, Any]]:
    if not isinstance(items, Sequence) or not items:
        return None
    filtered: List[Tuple[float, Dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        weight = float(item.get(weight_key, 0.0))
        if weight <= 0.0:
            continue
        filtered.append((weight, item))
    if not filtered:
        return None
    total = sum(weight for weight, _ in filtered)
    pick = rng.random() * total
    acc = 0.0
    for weight, item in filtered:
        acc += weight
        if pick <= acc:
            return item
    return filtered[-1][1]


def local_offset_to_world(dx_local: float, dy_local: float, yaw_rad: float) -> Vec2:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (c * dx_local - s * dy_local, s * dx_local + c * dy_local)


def sample_local_frame_delta(
    rng: random.Random,
    priors_by_category: Dict[str, Any],
    category: str,
) -> Tuple[float, float, float]:
    priors = priors_by_category.get(category) if isinstance(priors_by_category, dict) else None
    if not isinstance(priors, dict):
        priors = {}
    dx_local = clipped_gaussian(
        rng,
        float(priors.get("forward_backward_sigma_m", 0.0)),
        float(priors.get("forward_backward_clip_m", 0.0)),
    )
    dy_local = clipped_gaussian(
        rng,
        float(priors.get("lateral_sigma_m", 0.0)),
        float(priors.get("lateral_clip_m", 0.0)),
    )
    dtheta_deg = clipped_gaussian(
        rng,
        float(priors.get("yaw_sigma_deg", 0.0)),
        float(priors.get("yaw_clip_deg", 0.0)),
    )
    return dx_local, dy_local, dtheta_deg


def sample_multiple_object_ids(
    rng: random.Random,
    preferred_ids: Sequence[str],
    low_priority_ids: Sequence[str],
    count: int,
) -> List[str]:
    preferred = list(dict.fromkeys(str(x) for x in preferred_ids if str(x)))
    low_priority = [str(x) for x in low_priority_ids if str(x) and str(x) not in preferred]
    picked: List[str] = []
    pool = preferred + low_priority
    if count <= 0 or not pool:
        return picked

    preferred_set = set(preferred)
    while len(picked) < count and pool:
        weighted: List[Tuple[float, str]] = []
        for oid in pool:
            weight = 1.0 if oid in preferred_set else 0.35
            weighted.append((weight, oid))
        total = sum(weight for weight, _ in weighted)
        pick = rng.random() * total
        acc = 0.0
        chosen = weighted[-1][1]
        for weight, oid in weighted:
            acc += weight
            if pick <= acc:
                chosen = oid
                break
        picked.append(chosen)
        pool = [oid for oid in pool if oid != chosen]
    return picked
