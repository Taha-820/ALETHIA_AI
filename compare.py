#!/usr/bin/env python3
"""Score results against a reference with the three checks of section 5 of the brief.

    python compare.py reference.jsonl[.gz] results.jsonl[.gz]

Only (video, second) pairs present in the reference are scored; a second missing from the results counts as
no objects and 0 people. Object F1 is micro-averaged over (video, second, object) triples.
"""
import gzip, json, sys

THRESHOLDS = {"f1": 0.95, "exact": 0.85, "within1": 0.95}


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    secs, errors = {}, set()
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (set(d["objects"]), int(d["people"]))
    return secs, errors


def score(ref_path, res_path):
    ref, ref_err = load(ref_path)
    res, res_err = load(res_path)
    tp = fp = fn = exact = within1 = missing = 0
    per_class = {}
    for k, (robj, rp) in ref.items():
        if k not in res:
            missing += 1
        obj, p = res.get(k, (set(), 0))
        tp += len(obj & robj)
        fp += len(obj - robj)
        fn += len(robj - obj)
        for c in obj | robj:
            s = per_class.setdefault(c, [0, 0, 0])
            s[0] += c in obj and c in robj
            s[1] += c in obj and c not in robj
            s[2] += c in robj and c not in obj
        exact += p == rp
        within1 += abs(p - rp) <= 1
    n = max(1, len(ref))
    m = {"seconds": len(ref), "missing": missing, "extra": len(set(res) - set(ref)),
         "ref_errors": len(ref_err), "res_errors": len(res_err), "tp": tp, "fp": fp, "fn": fn,
         "f1": 2 * tp / max(1, 2 * tp + fp + fn), "exact": exact / n, "within1": within1 / n,
         "per_class": per_class}
    m["pass"] = all(m[k] >= v for k, v in THRESHOLDS.items())
    return m


def report(m):
    ok = lambda k: "PASS" if m[k] >= THRESHOLDS[k] else "FAIL"
    lines = [
        f"seconds in reference {m['seconds']}, missing from results {m['missing']}, extra in results {m['extra']}",
        f"error lines: reference {m['ref_errors']}, results {m['res_errors']}",
        f"object F1        {m['f1']:.4f}   (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})   need >= 0.95  {ok('f1')}",
        f"people exact     {m['exact']:.4f}   need >= 0.85  {ok('exact')}",
        f"people within 1  {m['within1']:.4f}   need >= 0.95  {ok('within1')}",
    ]
    for c, (a, b, d) in sorted(m["per_class"].items()):
        lines.append(f"  {c:11s} F1 {2 * a / max(1, 2 * a + b + d):.4f}  (tp {a}, fp {b}, fn {d})")
    return "\n".join(lines)


def main(ref_path, res_path):
    m = score(ref_path, res_path)
    print(report(m))
    return 0 if m["pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
