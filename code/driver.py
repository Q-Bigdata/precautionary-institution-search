
import json, os, random, time
import numpy as np
import search

def run_search(gen_fn, params, seeds, out_path, n_islands=4, children=8, generations=30, pop_k=12, r=0.4,
               families=("SR", "AL", "EV"), styles=(("improve", .4), ("combine", .25), ("simplify", .15), ("explore", .2)),
               n_workers=16, srcdir=None, rng_seed=0, migrate_every=5, zero_shot=False, log=print, cfg=None):
    rng = random.Random(rng_seed)
    seen = {}
    # score seeds
    items = [dict(id=f"seed_{k}", code=v) for k, v in seeds.items()]
    res = search.score_parallel(items, params, os.path.join(os.path.dirname(out_path) or ".", "work"), n_workers=n_workers,
                                r=r, families=families, srcdir=srcdir, cfg=cfg)
    allrecs = []
    for it, rs in zip(items, sorted(res, key=lambda x: [i["id"] for i in items].index(x["id"]))):
        rec = dict(rs, code=it["code"], gen=-1, island=-1, style="seed", parents=[])
        allrecs.append(rec); seen[search.code_hash(it["code"])] = rec["id"]
    good = [x for x in allrecs if x.get("ok")]
    islands = [sorted(good, key=lambda x: x["fit"]) for _ in range(n_islands)]
    archive = {}
    def add_archive(rec):
        nk = search.niche(rec)
        if nk not in archive or rec["fit"] > archive[nk]["fit"] or (rec["fit"] == archive[nk]["fit"] and rec["length"] < archive[nk]["length"]):
            archive[nk] = rec
    for x in good: add_archive(x)
    fout = open(out_path, "a")
    for x in allrecs: fout.write(json.dumps(x) + "\n")
    fout.flush()
    sty_names = [s for s, _ in styles]; sty_w = [w for _, w in styles]
    def tourney(pop, k=3):
        cand = rng.sample(pop, min(k, len(pop)))
        return max(cand, key=lambda x: (x["fit"], -x["length"]))
    for gen in range(generations):
        t0 = time.time()
        jobs = []
        for isl in range(n_islands):
            pop = islands[isl]
            for ch in range(children):
                if zero_shot:
                    style = rng.choices(sty_names, sty_w)[0]
                    parents = [x for x in allrecs if x["style"] == "seed" and x.get("ok")]
                else:
                    style = rng.choices(sty_names, sty_w)[0]
                    arch = list(archive.values())
                    if style == "improve":
                        parents = [tourney(pop), rng.choice(arch)]
                    elif style == "combine":
                        parents = [tourney(pop), tourney(pop), rng.choice(arch)]
                    elif style == "simplify":
                        parents = [max(pop, key=lambda x: x["fit"])]
                    else:
                        parents = rng.sample(arch, min(2, len(arch)))
                    uniq = {}
                    for pp in parents: uniq[pp["id"]] = pp
                    parents = sorted(uniq.values(), key=lambda x: x["fit"])
                single = len(families) == 1
                sty = "improve_single" if (single and style == "improve") else style
                jobs.append(dict(island=isl, style=style, parents=parents,
                                 prompt=search.build_prompt(parents, sty, g=r, single=single)))
        try:
            outs = gen_fn([j["prompt"] for j in jobs])
        except Exception as ex:
            log(f"LLM failure at gen {gen}: {str(ex)[:200]} -- stopping search")
            break
        items = []; meta = {}
        for j, o in zip(jobs, outs):
            code = search.extract_code(o.get("text", "")) if isinstance(o, dict) and "text" in o else None
            rid = f"g{gen}_i{j['island']}_{len(meta)}"
            base = dict(id=rid, gen=gen, island=j["island"], style=j["style"], parents=[p["id"] for p in j["parents"]],
                        out_tokens=(o.get("usage", {}) or {}).get("output_tokens") if isinstance(o, dict) else None)
            if code is None:
                meta[rid] = dict(base, ok=False, error="no code", code=None); continue
            h = search.code_hash(code)
            if h in seen:
                meta[rid] = dict(base, ok=False, error=f"duplicate of {seen[h]}", code=code); continue
            seen[h] = rid
            meta[rid] = dict(base, code=code); items.append(dict(id=rid, code=code))
        res = search.score_parallel(items, params, os.path.join(os.path.dirname(out_path) or ".", "work"),
                                    n_workers=n_workers, r=r, families=families, srcdir=srcdir, cfg=cfg)
        for rs in res:
            meta[rs["id"]].update(rs)
        new = list(meta.values())
        for x in new:
            fout.write(json.dumps(x) + "\n")
            allrecs.append(x)
            if x.get("ok"):
                add_archive(x)
                islands[x["island"]].append(x)
        for isl in range(n_islands):
            islands[isl] = sorted(islands[isl], key=lambda x: (x["fit"], -x["length"]))[-pop_k:]
        if migrate_every and (gen + 1) % migrate_every == 0 and not zero_shot:
            bests = [islands[i][-1] for i in range(n_islands)]
            for i in range(n_islands):
                islands[(i + 1) % n_islands].append(bests[i])
        fout.flush()
        okn = sum(1 for x in new if x.get("ok"))
        best = max((x for x in allrecs if x.get("ok")), key=lambda x: x["fit"])
        log(f"gen {gen}: {okn}/{len(new)} valid, gen-best {max([x['fit'] for x in new if x.get('ok')] or [0]):.3f}, "
            f"overall {best['fit']:.3f} ({best['id']}, len {best['length']}), niches {len(archive)}, {time.time()-t0:.0f}s")
    fout.close()
    return allrecs, archive, islands
