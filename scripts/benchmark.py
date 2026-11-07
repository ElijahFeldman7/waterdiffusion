import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WATERBENCH_DIR = PROJECT_ROOT / "alternative" / "zurich" / "data" / "WaterBench"
DATASETS = [
    ("2-1_longform_qa.jsonl",
     "You are a helpful assistant, please answer the following question within 300 words:\n{context}\n{input}"),
    ("2-2_finance_qa.jsonl",
     "You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n{context}\n{input}"),
    ("5-1_alpacafarm.jsonl",
     "You are a helpful assistant, please answer the following instruction: \n{context}\n{input}"),
]


def parse_method(spec: str) -> dict:
    name, _, rest = spec.partition(":")
    kv = {}
    if rest:
        for part in rest.split(","):
            k, v = part.split("=")
            kv[k.strip()] = float(v)
    if name == "none":
        return {"name": spec, "kind": "none", "delta": 0.0, "order": 0.0}
    if name == "dgmark":
        return {"name": spec, "kind": "dgmark", "delta": 0.0, "order": 0.0}
    if name == "zurich":
        return {"name": spec, "kind": "joint", "delta": kv["delta"], "order": 0.0}
    if name == "order":
        return {"name": spec, "kind": "joint", "delta": 0.0, "order": kv["order"]}
    if name == "joint":
        return {"name": spec, "kind": "joint", "delta": kv["delta"], "order": kv["order"]}
    raise ValueError(f"Unknown method spec {spec}")


def family(m: dict) -> str:
    if m["kind"] in ("none", "dgmark"):
        return m["kind"]
    if m["order"] == 0:
        return "zurich"
    if m["delta"] == 0:
        return "order"
    return "joint"


def load_prompts(num_samples: int):
    per_file = []
    for fname, template in DATASETS:
        rows = []
        with open(WATERBENCH_DIR / fname) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                prompt = template.format(context=ex.get("context", ""), input=ex.get("input", "")).strip()
                rows.append({"prompt": prompt, "source": fname, "pid": f"{fname}:{i}"})
        per_file.append(rows)
    out = []
    for i in range(max(len(r) for r in per_file)):
        for rows in per_file:
            if i < len(rows):
                out.append(rows[i])
    return out[:num_samples]


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def read_rows(run_dir: Path):
    rows = []
    for f in sorted(run_dir.glob("gen_shard_*.jsonl")):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def cmd_generate(args):
    from transformers import AutoModel, AutoTokenizer
    from watermarks.watermark import DgMark, JointWatermark, generate, trim_response

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    methods = [parse_method(s) for s in args.methods]
    assert any(m["kind"] == "none" for m in methods), "include 'none' (null distribution + PPL baseline)"

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({**vars(args), "func": None,
                                        "timestamp_utc": datetime.now(timezone.utc).isoformat()}, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    mask_id = tok.mask_token_id if getattr(tok, "mask_token_id", None) is not None else 126336
    vocab_size = len(tok.get_vocab())
    stop_ids = {mask_id, 126081, 126348}
    if tok.eos_token_id is not None:
        stop_ids.add(tok.eos_token_id)
    for t in ("<|eot_id|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            stop_ids.add(tid)

    # All JointWatermark instances share Zurich's fixed-seed green list, so one detector serves all.
    detector = JointWatermark(vocab_size, delta=0.0, order_strength=0.0, gamma=args.gamma, device=device,
                              mask_token_id=mask_id)
    dg = DgMark(private_key=args.dg_key)
    wms = {m["name"]: JointWatermark(vocab_size, delta=m["delta"], order_strength=m["order"], gamma=args.gamma,
                                     belief_topk=args.belief_topk, device=device, mask_token_id=mask_id)
           for m in methods if m["kind"] == "joint"}

    prompts = load_prompts(args.num_samples)
    prompts = prompts[args.shard::args.num_shards]
    out_path = run_dir / f"gen_shard_{args.shard}_of_{args.num_shards}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    done.add((r["pid"], r["method"]))

    print(f"[shard {args.shard}] {len(prompts)} prompts x {len(methods)} methods; {len(done)} done", flush=True)
    with open(out_path, "a") as fout:
        for pi, pr in enumerate(prompts):
            chat = tok.apply_chat_template([{"role": "user", "content": pr["prompt"]}],
                                           add_generation_prompt=True, tokenize=False)
            prompt_ids = tok(chat, return_tensors="pt")["input_ids"].to(device)
            P = prompt_ids.shape[1]
            for m in methods:
                if (pr["pid"], m["name"]) in done:
                    continue
                torch.manual_seed(stable_seed(args.seed, pr["pid"]))
                t0 = time.time()
                x, diag = generate(
                    model, prompt_ids, kind=m["kind"], wm=wms.get(m["name"]), dg=dg,
                    steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
                    temperature=args.temperature, remasking=args.remasking, sched_tau=args.sched_tau,
                    mask_id=mask_id,
                )
                gen_time = time.time() - t0
                resp = trim_response(x[0, P:], sorted(stop_ids))
                text = tok.decode(resp, skip_special_tokens=True)
                d_green = detector.detect_ids(resp)
                d_dg = dg.detect_ids(resp.cpu(), P)
                nc = max(diag["n_committed"], 1)
                row = {
                    "pid": pr["pid"], "source": pr["source"], "method": m["name"], "family": family(m),
                    "kind": m["kind"], "delta": m["delta"], "order": m["order"],
                    "prompt": pr["prompt"], "completion": text, "n_tokens": int(resp.numel()),
                    "p_green": d_green["p_value"], "z_green": d_green["z_score"],
                    "green_rate": d_green["green_rate"], "n_scored_green": d_green["n_scored"],
                    "p_dg": d_dg["p_value"], "z_dg": d_dg["z_score"], "dg_match_rate": d_dg["match_rate"],
                    "token_kl_per_tok": diag["token_kl_sum"] / nc,
                    "sel_kl_per_step": diag["sel_kl_sum"] / max(diag["n_steps"], 1),
                    "frac_order_changed": diag["order_changed"] / nc,
                    "gain_per_committed": diag["gain_committed_sum"] / nc,
                    "gen_time": gen_time,
                    "temperature": args.temperature, "remasking": args.remasking, "sched_tau": args.sched_tau,
                    "gen_length": args.gen_length, "steps": args.steps, "block_length": args.block_length,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
            print(f"[shard {args.shard}] {pi + 1}/{len(prompts)}", flush=True)


def cmd_ppl(args):
    from torch.nn import CrossEntropyLoss
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run_dir)
    rows = read_rows(run_dir)
    tag = args.ppl_model.replace("/", "_")
    out_path = run_dir / f"ppl_{tag}.jsonl"
    have = set()
    if out_path.exists():
        with open(out_path) as f:
            have = {(r["pid"], r["method"]) for r in (json.loads(l) for l in f if l.strip())}
    todo = [r for r in rows if (r["pid"], r["method"]) not in have and r["completion"].strip()]
    print(f"PPL: {len(todo)} rows to score with {args.ppl_model}", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ppl_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.ppl_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    loss_fct = CrossEntropyLoss(reduction="none")

    with open(out_path, "a") as fout:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            prompts = [r["prompt"] for r in batch]
            full = [p + c for p, c in zip(prompts, [r["completion"] for r in batch])]
            enc = tok(full, add_special_tokens=True, padding=True, return_tensors="pt").to(model.device)
            plen = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in prompts]
            with torch.no_grad():
                logits = model(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            score_mask = enc["attention_mask"].clone()
            for b, L in enumerate(plen):
                score_mask[b, :L] = 0
            sl = logits[:, :-1].float()
            lab = enc["input_ids"][:, 1:]
            sm = score_mask[:, 1:].float()
            nll = (loss_fct(sl.transpose(1, 2), lab) * sm).sum(1) / sm.sum(1).clamp_min(1)
            for r, v in zip(batch, nll.tolist()):
                fout.write(json.dumps({"pid": r["pid"], "method": r["method"], "log_ppl": v}) + "\n")
            fout.flush()
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)


def tpr_at_fpr(null_p: np.ndarray, wm_p: np.ndarray, fpr: float = 0.01):
    s = np.sort(null_p)
    k = int(math.floor(fpr * len(s)))
    if k == 0:
        return float("nan"), float("nan")
    t = s[k]
    return float(np.mean(wm_p < t)), float(np.mean(null_p < t))


def bootstrap_ci(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def cmd_summarize(args):
    import pandas as pd

    run_dir = Path(args.run_dir)
    df = pd.DataFrame(read_rows(run_dir))
    ppl_files = sorted(run_dir.glob("ppl_*.jsonl"))
    if args.ppl_file:
        ppl_files = [run_dir / args.ppl_file]
    if ppl_files:
        ppl = pd.concat([pd.read_json(f, lines=True) for f in ppl_files]).drop_duplicates(["pid", "method"], keep="last")
        df = df.merge(ppl, on=["pid", "method"], how="left")
        print(f"Using PPL from: {[f.name for f in ppl_files]}")
    else:
        df["log_ppl"] = np.nan
        print("WARNING: no PPL file found; run the `ppl` stage first. Reporting detection only.")

    df = df[df["n_tokens"] >= args.min_tokens].copy()
    base = df[df["kind"] == "none"].set_index("pid")
    null_green = base["p_green"].to_numpy()
    null_dg = base["p_dg"].to_numpy()
    print(f"Null samples (none, >= {args.min_tokens} tokens): {len(base)}")
    if len(base) < 200:
        print("WARNING: fewer than 200 null samples -> the 1% FPR threshold rests on <2 points.")

    recs = []
    for name, g in df.groupby("method"):
        fam = g["family"].iloc[0]
        col = "p_dg" if fam == "dgmark" else "p_green"
        null = null_dg if fam == "dgmark" else null_green
        wm_p = g[col].to_numpy()
        tpr, fpr_emp = tpr_at_fpr(null, wm_p) if fam != "none" else (float("nan"), float("nan"))
        tpr_ci = bootstrap_ci(lambda idx: tpr_at_fpr(null, wm_p[idx])[0], len(wm_p)) if fam != "none" else (np.nan, np.nan)

        gp = g.set_index("pid")
        common = gp.index.intersection(base.index)
        dlog = (gp.loc[common, "log_ppl"] - base.loc[common, "log_ppl"]).to_numpy()
        dlog = dlog[np.isfinite(dlog)]
        dlog_ci = bootstrap_ci(lambda idx: float(np.mean(dlog[idx])), len(dlog)) if len(dlog) else (np.nan, np.nan)

        recs.append({
            "method": name, "family": fam, "delta": g["delta"].iloc[0], "order": g["order"].iloc[0], "n": len(g),
            "TPR@1%FPR": tpr, "TPR_lo": tpr_ci[0], "TPR_hi": tpr_ci[1], "emp_FPR": fpr_emp,
            "TPR@p<.01(nominal)": float(np.mean(wm_p < 0.01)) if fam != "none" else float("nan"),
            "null_FPR@p<.01(nominal)": float(np.mean(null < 0.01)),
            "log_ppl": float(g["log_ppl"].mean()),
            "dlog_ppl": float(np.mean(dlog)) if len(dlog) else float("nan"),
            "dlog_lo": dlog_ci[0], "dlog_hi": dlog_ci[1],
            "green_rate": float(g["green_rate"].mean()), "dg_match_rate": float(g["dg_match_rate"].mean()),
            "token_kl/tok": float(g["token_kl_per_tok"].mean()), "sel_kl/step": float(g["sel_kl_per_step"].mean()),
            "frac_order_changed": float(g["frac_order_changed"].mean()),
            "mean_tokens": float(g["n_tokens"].mean()), "gen_time_s": float(g["gen_time"].mean()),
        })
    res = pd.DataFrame(recs).sort_values(["family", "delta", "order"])
    res.to_csv(run_dir / "summary.csv", index=False)
    with open(run_dir / "summary.md", "w") as f:
        try:
            f.write(res.to_markdown(index=False, floatfmt=".3f"))
        except ImportError:  # tabulate not installed
            f.write(res.to_string(index=False))
    pd.set_option("display.width", 250)
    print(res[["method", "n", "TPR@1%FPR", "TPR_lo", "TPR_hi", "dlog_ppl", "dlog_lo", "dlog_hi",
               "green_rate", "token_kl/tok", "frac_order_changed", "gen_time_s"]].to_string(index=False))

    # plot TPR vs dlog PPL
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"zurich": "#1f77b4", "order": "#ff7f0e", "joint": "#2ca02c", "dgmark": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xcol, xlabel in [(axes[0], "dlog_ppl", r"$\Delta \log$ PPL (paired vs. none)"),
                             (axes[1], "log_ppl", r"$\log$ PPL")]:
        for fam, g in res[res["family"] != "none"].groupby("family"):
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g["TPR@1%FPR"], "-o" if fam in ("zurich", "order") else "o",
                    color=colors.get(fam, "k"), label=fam)
            for _, r in g.iterrows():
                ax.annotate(r["method"].split(":")[-1], (r[xcol], r["TPR@1%FPR"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            if xcol == "dlog_ppl":
                ax.errorbar(g[xcol], g["TPR@1%FPR"],
                            xerr=[g[xcol] - g["dlog_lo"], g["dlog_hi"] - g[xcol]],
                            yerr=[g["TPR@1%FPR"] - g["TPR_lo"], g["TPR_hi"] - g["TPR@1%FPR"]],
                            fmt="none", ecolor=colors.get(fam, "k"), alpha=0.3)
        if xcol == "log_ppl" and len(res[res["family"] == "none"]):
            ax.axvline(res[res["family"] == "none"]["log_ppl"].iloc[0], color="gray", ls="--", label="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TPR @ 1% FPR (empirical)")
        ax.grid(alpha=0.3)
        ax.legend()
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    fig.suptitle(f"LLaDA WaterBench | T={cfg.get('temperature')} remask={cfg.get('remasking')} "
                 f"sched_tau={cfg.get('sched_tau')} | n_null={len(base)}")
    fig.tight_layout()
    fig.savefig(run_dir / "tpr_vs_ppl.png", dpi=200)
    print(f"Wrote {run_dir/'summary.csv'}, {run_dir/'summary.md'}, {run_dir/'tpr_vs_ppl.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--run_dir", required=True)
    g.add_argument("--methods", nargs="+", required=True)
    g.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    g.add_argument("--num_samples", type=int, default=600)
    g.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    g.add_argument("--num_shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    g.add_argument("--gen_length", type=int, default=300)
    g.add_argument("--steps", type=int, default=300)
    g.add_argument("--block_length", type=int, default=25)
    g.add_argument("--temperature", type=float, default=0.5)
    g.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    g.add_argument("--sched_tau", type=float, default=0.0,
                   help="Plackett-Luce scheduler temperature (0 = standard deterministic top-k, applied to ALL methods)")
    g.add_argument("--gamma", type=float, default=0.25)
    g.add_argument("--belief_topk", type=int, default=50)
    g.add_argument("--dg_key", default=None, help="dgMARK private key (keyed rule degenerates to 'token even'; see watermark.py)")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("ppl")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ppl_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--batch_size", type=int, default=4)
    p.set_defaults(func=cmd_ppl)

    s = sub.add_parser("summarize")
    s.add_argument("--run_dir", required=True)
    s.add_argument("--min_tokens", type=int, default=100)
    s.add_argument("--ppl_file", default=None)
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
