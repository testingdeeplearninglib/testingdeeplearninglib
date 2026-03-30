import json, collections, sys

def top_counts(list_of_lists, k=12):
    c = collections.Counter()
    for xs in list_of_lists:
        for x in (xs or []):
            c[str(x)] += 1
    return [x for x, _ in c.most_common(k)]

def main():
    if len(sys.argv) != 3:
        print("USAGE: python make_prompt_pack.py <GCFL_JSON> <PROMPT_PACK_JSON>")
        raise SystemExit(2)

    p = sys.argv[1]
    outp = sys.argv[2]
    gcfl = json.load(open(p, "r", encoding="utf-8"))

    packs = []
    for c in gcfl:
        members = c.get("members", [])
        if not members:
            continue

        scenario = c.get("scenario", "OTHER")
        size = int(c.get("size", len(members)))

        libs = collections.Counter((m.get("library") or "NA") for m in members)
        top_lib, top_cnt = libs.most_common(1)[0]
        purity = (top_cnt / size) if size else 0.0

        oracle_union = sorted({t for m in members for t in (m.get("oracle_types") or [])})
        fam_top = top_counts([m.get("families") for m in members], k=15)
        kw_top  = top_counts([m.get("keywords") for m in members], k=20)

        ev = []
        for m in members[:12]:
            ev.append({
                "library": m.get("library"),
                "bug_no": m.get("bug_no"),
                "title": m.get("title"),
                "url": m.get("url"),
            })

        packs.append({
            "gcfl_id": c.get("gcfl_id"),
            "scenario_type": scenario,
            "cluster_size": size,
            "library_purity": round(purity, 3),
            "top_library": top_lib,
            "oracle_types": oracle_union,
            "top_families": fam_top,
            "top_keywords": kw_top,
            "evidence_members": ev,
        })

    json.dump(packs, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("WROTE", outp, "clusters=", len(packs))

if __name__ == "__main__":
    main()
