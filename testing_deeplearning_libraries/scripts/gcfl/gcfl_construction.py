#!/usr/bin/env python3
# GCFL_v12.py (v12.5)
#
# Fixes vs v12.4:
# 1) If TAG shared >= tag_min_shared but lacks a "strong" tag, we DO NOT return False.
#    We just skip TAG-linking and fall through to keyword similarity.
# 2) require_shared_oracle_type does NOT block hard-id links (PRE/TRIG/ORID).
# 3) Keyword similarity defaults are still configurable; your previous values were too strict.

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


SCENARIOS = {
    "OTHER",
    "TVM_RELAY_CODEGEN",
    "SERIALIZATION_CHECKPOINTING",
    "TRACING_GRAPH_FUNCTION",
    "TRAINING_FIT_EVALUATE",
    "DTYPE_PRECISION",
    "DATA_PIPELINE",
    "DISTRIBUTED",
    "AUTOGRAD_BACKWARD",
    "SPARSE",
}

ORACLE_TYPES = {
    "exception",
    "crash",
    "output_mismatch",
    "logging_based",
    "grad_mismatch",
    "mode_mismatch",
    "serialization_mismatch",
    "nondeterminism",
    "backend_mismatch",
}

DEFAULT_STOPWORDS = {
    "python", "error", "errors", "issue", "bug", "crash", "fails", "failure",
    "model", "models", "layer", "layers", "ops", "op", "api",
    "train", "training", "eval", "evaluate", "fit",
    "tensor", "tensors", "dtype", "shape", "device",
    "gpu", "gpus", "cpu",
    "exception", "runtime", "value", "values",
    "works", "work", "doesn", "doesnt", "doesn't",
    "not", "with", "without", "when", "where", "after", "before", "into", "from",
}

_word_re = re.compile(r"[a-z0-9_]+")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def norm_token(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def extract_words(text: str) -> List[str]:
    text = str(text).lower().replace("-", "_")
    return _word_re.findall(text)


def pick_scenario(entry: Dict[str, Any]) -> str:
    sc = entry.get("scenario")
    if isinstance(sc, str) and sc in SCENARIOS:
        return sc

    tags = entry.get("scope_tags") or entry.get("tags") or []
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str) and t in SCENARIOS:
                return t

    bm = entry.get("bug_meta") or {}
    if isinstance(bm, dict):
        sc2 = bm.get("scenario")
        if isinstance(sc2, str) and sc2 in SCENARIOS:
            return sc2

    return "OTHER"


def pick_library(entry: Dict[str, Any]) -> str:
    if isinstance(entry.get("library"), str) and entry["library"].strip():
        return entry["library"].strip()
    bm = entry.get("bug_meta") or {}
    if isinstance(bm, dict) and isinstance(bm.get("library"), str):
        return bm["library"].strip()
    return "unknown"


def pick_bug_no(entry: Dict[str, Any]) -> Optional[int]:
    for k in ("bug_no", "bug_number", "issue_no", "issue_number"):
        v = entry.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    bm = entry.get("bug_meta") or {}
    if isinstance(bm, dict):
        v = bm.get("bug_no") or bm.get("issue_number")
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def pick_title(entry: Dict[str, Any]) -> str:
    if isinstance(entry.get("title"), str):
        return entry["title"]
    bm = entry.get("bug_meta") or {}
    if isinstance(bm, dict) and isinstance(bm.get("title"), str):
        return bm["title"]
    return ""


def pick_url(entry: Dict[str, Any]) -> str:
    if isinstance(entry.get("url"), str):
        return entry["url"]
    bm = entry.get("bug_meta") or {}
    if isinstance(bm, dict) and isinstance(bm.get("url"), str):
        return bm["url"]
    return ""


def extract_oracle_types(entry: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()

    oracle_obj = entry.get("oracle")
    if isinstance(oracle_obj, list):
        for o in oracle_obj:
            if isinstance(o, dict):
                t = o.get("type")
                if isinstance(t, str):
                    tt = norm_token(t)
                    if tt in ORACLE_TYPES:
                        out.add(tt)

    oracles = entry.get("oracles")
    if isinstance(oracles, list):
        for o in oracles:
            if isinstance(o, str):
                tt = norm_token(o)
                if tt in ORACLE_TYPES:
                    out.add(tt)

    po = entry.get("primary_oracle")
    if isinstance(po, str):
        tt = norm_token(po)
        if tt in ORACLE_TYPES:
            out.add(tt)

    return out or {"exception"}


def derive_tag_families(entry: Dict[str, Any]) -> Set[str]:
    tags = entry.get("scope_tags") or []
    fams: Set[str] = set()
    if not isinstance(tags, list):
        return fams
    for t in tags:
        if not isinstance(t, str) or not t.strip():
            continue
        if t in SCENARIOS:
            continue
        fams.add("CFAM:TAG_" + norm_token(t))
    return fams


def derive_structured_families(entry: Dict[str, Any]) -> Set[str]:
    fams: Set[str] = set()

    pre = entry.get("preconditions")
    if isinstance(pre, list):
        for x in pre:
            if isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"].strip():
                fams.add("CFAM:PRE_" + norm_token(x["id"]))

    trig = entry.get("trigger_features")
    if isinstance(trig, list):
        for x in trig:
            if isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"].strip():
                fams.add("CFAM:TRIG_" + norm_token(x["id"]))

    oracle = entry.get("oracle")
    if isinstance(oracle, list):
        for x in oracle:
            if isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"].strip():
                fams.add("CFAM:ORID_" + norm_token(x["id"]))

    return fams


def extract_keywords(entry: Dict[str, Any], topk: int) -> List[str]:
    if topk <= 0:
        return []

    kws = entry.get("keywords_sample") or entry.get("keywords") or []
    words: List[str] = []

    if isinstance(kws, list) and kws:
        for w in kws:
            if isinstance(w, str):
                words.extend(extract_words(w))
    else:
        title = pick_title(entry)
        nature = entry.get("nature") or ""
        words.extend(extract_words(title))
        words.extend(extract_words(nature))

    normed: List[str] = []
    for w in words:
        w = norm_token(w)
        if not w or w.isdigit() or len(w) <= 2:
            continue
        normed.append(w)

    c = Counter(normed)
    ranked = [w for w, _ in c.most_common(topk * 3)]

    out: List[str] = []
    seen: Set[str] = set()
    for w in ranked:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= topk:
            break
    return out


def extract_nature_tokens(entry: Dict[str, Any], enabled: bool) -> Set[str]:
    if not enabled:
        return set()
    nat = entry.get("nature")
    if not isinstance(nat, str) or not nat.strip():
        return set()
    t = norm_token(nat)
    toks = {f"nature:{t}"}
    for w in extract_words(nat):
        ww = norm_token(w)
        if ww and len(ww) > 2:
            toks.add(f"natw:{ww}")
    return toks


@dataclass
class Item:
    idx: int
    library: str
    bug_no: Optional[int]
    scenario: str
    title: str
    url: str
    oracle_types: Set[str]
    families: Set[str]
    tags: Set[str]
    hard: Set[str]
    keywords: List[str]
    nature_tokens: Set[str]


def df_filter_tag_families(items: List[Item], tag_df_ratio: float) -> None:
    n = len(items)
    if n == 0:
        return
    df = Counter()
    for it in items:
        for t in it.tags:
            df[t] += 1

    drop = {t for t, c in df.items() if (c / n) >= tag_df_ratio}
    if not drop:
        return

    for it in items:
        it.tags = {t for t in it.tags if t not in drop}
        it.families = {f for f in it.families if f not in drop}


def df_filter_structured_families(items: List[Item], structured_df_ratio: float) -> None:
    n = len(items)
    if n == 0:
        return
    df = Counter()
    for it in items:
        for h in it.hard:
            df[h] += 1

    drop = {h for h, c in df.items() if (c / n) >= structured_df_ratio}
    if not drop:
        return

    for it in items:
        it.hard = {h for h in it.hard if h not in drop}
        it.families = {f for f in it.families if f not in drop}


def df_filter_keywords(items: List[Item], stopwords: Set[str], high_df_ratio: float) -> None:
    by_scn: Dict[str, List[Item]] = defaultdict(list)
    for it in items:
        by_scn[it.scenario].append(it)

    for scn, group in by_scn.items():
        n = len(group)
        if n <= 0:
            continue

        df = Counter()
        for it in group:
            toks = set(it.keywords)
            for t in it.nature_tokens:
                if t.startswith("natw:"):
                    toks.add(t[len("natw:"):])
            for t in toks:
                df[t] += 1

        drop = set()
        for t, c in df.items():
            if t in stopwords:
                drop.add(t)
                continue
            if (c / n) >= high_df_ratio:
                drop.add(t)

        if not drop:
            continue

        for it in group:
            it.keywords = [k for k in it.keywords if k not in drop]
            it.nature_tokens = {
                t for t in it.nature_tokens
                if not (t.startswith("natw:") and t[len("natw:"):] in drop)
            }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def shared_oracle_type(a: Item, b: Item) -> bool:
    return bool(a.oracle_types & b.oracle_types)


def keyword_token_set(it: Item) -> Set[str]:
    s = {f"kw:{k}" for k in it.keywords}
    s |= set(it.nature_tokens)
    return s


def strong_tag_name(tag_fam: str) -> str:
    if not tag_fam.startswith("CFAM:TAG_"):
        return ""
    return tag_fam[len("CFAM:TAG_"):]


def should_link(a: Item, b: Item, args: argparse.Namespace) -> bool:
    if args.within_scenario and a.scenario != b.scenario:
        return False
    if args.within_library and a.library != b.library:
        return False

    # 1) HARD link first (must not be blocked by oracle gating)
    if args.min_hard_family_shared > 0:
        if len(a.hard & b.hard) >= args.min_hard_family_shared:
            return True

    # 2) Oracle gating applies only after hard-link check
    if args.require_shared_oracle_type and not shared_oracle_type(a, b):
        return False

    # 3) TAG edge (but do NOT early-return False if strong requirement fails)
    if args.use_tag_edges:
        shared_tags = a.tags & b.tags
        if len(shared_tags) >= args.tag_min_shared:
            if args.tag_require_strong:
                strong_hit = any(
                    (strong_tag_name(tfam) in args._strong_tag_allow)
                    for tfam in shared_tags
                )
                if strong_hit:
                    return True
                # No strong tag -> skip TAG edge, continue to keywords
            else:
                return True

    # Plan A stops here
    if args.plan == "A":
        return False

    # 4) Keyword similarity fallback
    ta = keyword_token_set(a)
    tb = keyword_token_set(b)
    shared = len(ta & tb)
    if shared < args.min_shared:
        return False
    return jaccard(ta, tb) >= args.jaccard


def cluster(items: List[Item], args: argparse.Namespace) -> Dict[int, List[Item]]:
    n = len(items)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if should_link(items[i], items[j], args):
                dsu.union(i, j)

    comps: Dict[int, List[Item]] = defaultdict(list)
    for i, it in enumerate(items):
        comps[dsu.find(i)].append(it)
    return comps


def build_items(raw: Any, args: argparse.Namespace) -> List[Item]:
    if isinstance(raw, dict) and "entries" in raw and isinstance(raw["entries"], list):
        entries = raw["entries"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("Unsupported input JSON format: expected list or {entries:[...]}")

    items: List[Item] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue

        scn = pick_scenario(e)
        lib = pick_library(e)
        bno = pick_bug_no(e)
        title = pick_title(e)
        url = pick_url(e)
        or_types = extract_oracle_types(e)

        fams: Set[str] = set()
        tags: Set[str] = set()
        hard: Set[str] = set()

        if args.derive_families:
            tags = derive_tag_families(e)
            fams |= tags

        if args.use_structured_ids:
            hard = derive_structured_families(e)
            fams |= hard

        kws = extract_keywords(e, args.keyword_topk)
        nat = extract_nature_tokens(e, enabled=(args.plan == "C"))

        items.append(
            Item(
                idx=i,
                library=lib,
                bug_no=bno,
                scenario=scn,
                title=title,
                url=url,
                oracle_types=or_types,
                families=fams,
                tags=set(tags),
                hard=set(hard),
                keywords=kws,
                nature_tokens=nat,
            )
        )
    return items


def build_outputs(comps: Dict[int, List[Item]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    clusters = sorted(comps.values(), key=lambda v: (-len(v), v[0].idx if v else 10**9))

    out_clusters: List[Dict[str, Any]] = []
    unmapped_entries: List[Dict[str, Any]] = []

    for ci, members in enumerate(clusters, start=1):
        scs = Counter([m.scenario for m in members])
        scenario = scs.most_common(1)[0][0] if scs else "OTHER"
        cid = f"GCFL-{scenario}-{ci:04d}"

        fam_union = sorted(set().union(*[m.families for m in members]) if members else [])
        oracle_union = sorted(set().union(*[m.oracle_types for m in members]) if members else [])

        mlist = []
        for m in members:
            mlist.append(
                {
                    "library": m.library,
                    "bug_no": m.bug_no,
                    "scenario": m.scenario,
                    "title": m.title,
                    "url": m.url,
                    "oracle_types": sorted(list(m.oracle_types)),
                    "families": sorted(list(m.families)),
                    "keywords": list(m.keywords),
                }
            )
            if not m.hard:
                unmapped_entries.append(
                    {
                        "library": m.library,
                        "bug_no": m.bug_no,
                        "title": m.title,
                        "scenario": m.scenario,
                        "oracle_types": sorted(list(m.oracle_types)),
                        "families": sorted(list(m.families)),
                    }
                )

        out_clusters.append(
            {
                "gcfl_id": cid,
                "scenario": scenario,
                "size": len(members),
                "families_union": fam_union,
                "oracle_types_union": oracle_union,
                "members": mlist,
            }
        )

    sizes = sorted([c["size"] for c in out_clusters], reverse=True)
    singleton = sum(1 for s in sizes if s == 1)
    scenario_dist = Counter()
    for c in out_clusters:
        scenario_dist[c["scenario"]] += c["size"]

    diagnostics = {
        "clusters": len(out_clusters),
        "top_cluster_sizes": sizes[:15],
        "singleton_clusters": singleton,
        "scenario_distribution": dict(scenario_dist),
    }

    unmapped_report = {
        "unmapped_total": len(unmapped_entries),
        "unmapped_entries": unmapped_entries,
    }

    return out_clusters, diagnostics, unmapped_report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--diagnostics", required=True)
    ap.add_argument("--unmapped_report", required=True)

    ap.add_argument("--plan", choices=["A", "B", "C"], default="B")

    ap.add_argument("--within_scenario", type=int, default=1)
    ap.add_argument("--within_library", type=int, default=0)
    ap.add_argument("--require_shared_oracle_type", type=int, default=0)

    ap.add_argument("--derive_families", type=int, default=1)
    ap.add_argument("--use_structured_ids", type=int, default=1)

    ap.add_argument("--min_hard_family_shared", type=int, default=1)

    ap.add_argument("--use_tag_edges", type=int, default=1)
    ap.add_argument("--tag_min_shared", type=int, default=2)
    ap.add_argument("--tag_df_ratio", type=float, default=0.20)
    ap.add_argument("--tag_require_strong", type=int, default=1)

    ap.add_argument("--keyword_topk", type=int, default=18)
    ap.add_argument("--min_shared", type=int, default=2)
    ap.add_argument("--jaccard", type=float, default=0.10)
    ap.add_argument("--high_df_ratio", type=float, default=0.85)

    ap.add_argument("--structured_df_ratio", type=float, default=0.40)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    args.within_scenario = bool(int(args.within_scenario))
    args.within_library = bool(int(args.within_library))
    args.require_shared_oracle_type = bool(int(args.require_shared_oracle_type))
    args.derive_families = bool(int(args.derive_families))
    args.use_structured_ids = bool(int(args.use_structured_ids))
    args.use_tag_edges = bool(int(args.use_tag_edges))
    args.tag_require_strong = bool(int(args.tag_require_strong))

    # Strong tags: keep broad glue out (keras/gpu/cpu/relay intentionally not strong)
    args._strong_tag_allow = {
        "tf_function",
        "gradient_tape",
        "tensorarray",
        "control_flow",
        "mirrored_strategy",
        "tpu",
        "tflite",
        "onnx_import",
        "meta_schedule",
        "tir",
        "shape_inference",
        "dynamic_shape",
        "eager_mode",
        "tf1_graph_mode",
        "autodiff",
        "serialization",
        "saved_model",
        "savedmodel",
        "tfrecord",
        "tf_data",
        "grappler",
        "post_training_quantization",
        "keras_3",
        "conv2d",
        "rnn",
        "tf_backend",
        "tf_py_function",
        "tf_estimator",
        "optimizer_apply_gradients",
        "model_fit",
        "distributed_training",
        "multi_gpu",
        "inference",
        "gradient",
        "tf_keras",
    }

    raw = load_json(args.input)
    items = build_items(raw, args)

    # DF filters
    if args.derive_families:
        df_filter_tag_families(items, tag_df_ratio=args.tag_df_ratio)
    if args.use_structured_ids:
        df_filter_structured_families(items, structured_df_ratio=args.structured_df_ratio)
    df_filter_keywords(items, DEFAULT_STOPWORDS, high_df_ratio=args.high_df_ratio)

    comps = cluster(items, args)
    clusters_out, diagnostics, unmapped = build_outputs(comps)

    sizes = sorted([c["size"] for c in clusters_out], reverse=True)
    singleton = sum(1 for s in sizes if s == 1)

    print(f"=== GCFL Build Summary (v12.5 Plan {args.plan}) ===")
    print(f"Input entries: {len(items)}")
    print(f"GCFL entries (clusters): {len(clusters_out)}")
    print(f"Top cluster sizes: {sizes[:15]}")
    print(f"Singleton clusters: {singleton} / {len(clusters_out)}")
    print("Scenario distribution:")
    for k, v in sorted(diagnostics["scenario_distribution"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

    dump_json(args.output, clusters_out)
    dump_json(args.diagnostics, diagnostics)
    dump_json(args.unmapped_report, unmapped)


if __name__ == "__main__":
    main()
