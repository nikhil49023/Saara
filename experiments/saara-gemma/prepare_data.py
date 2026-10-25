"""Build the saara-gemma dataset from public sources: format, filter, group-split, dedup, manifest.

Usage:
  python prepare_data.py --out data            # full build
  python prepare_data.py --out data_smoke --smoke

Output: <out>/{train,val,test}.jsonl, <out>/images/, <out>/data_manifest.json
Rules (spec §3-4): split by group (never by row); test frozen and hashed; near-duplicates of
test rows removed from train/val; every row carries provenance and license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tarfile
from ast import literal_eval as parse_python_literal  # literals only, never executes code
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from augment import degrade
from clutter import add_clutter
from prompts import CLEAN_USER, OCR_USER, PROMPT_VERSION, SYSTEM

SEED = 20261025
MAX_SIDE = 2048
RENDER_DPI = 150  # must equal mlforge.ingest.pdf PDFIngestConfig.dpi (train/inference parity)
LICENSES = {
    "allenai/olmOCR-mix-1025": "odc-by",
    "PleIAs/Post-OCR-Correction": "cc0-1.0",
    "HuggingFaceFW/fineweb-edu": "odc-by",
    "Agent-Ark/Toucan-1.5M": "apache-2.0",
    "NousResearch/hermes-function-calling-v1": "apache-2.0",
    "nvidia/When2Call": "cc-by-4.0",
    "allenai/tulu-3-sft-mixture": "odc-by",
}
# Historical sources (handwritten LoC transcripts, national archives) are up-weighted on purpose.
# (train pages, train tarballs to read, test pages, share of train pages that also get a degraded copy)
OCR_PLAN = {
    "00_documents": (2000, 1, 200, 0.3),
    "01_books": (1500, 1, 200, 0.3),
    "02_loc_transcripts": (1000, 1, 98, 0.5),
    "03_national_archives": (1000, 2, 169, 0.5),
}
DEFAULT_CAPS = {"post_ocr": 3500, "clutter": 2500, "toucan": 6000, "hermes": 1500, "when2call": 1500,
                "when2call_test": 400, "replay": 1500, "noisy_test_share": 0.5}
SMOKE_CAPS = {k: (0.5 if k == "noisy_test_share" else 6) for k in DEFAULT_CAPS}
drops: Counter[str] = Counter()


# ---------- helpers ----------

def split_of(group: str, val: float = 0.05, test: float = 0.05) -> str:
    bucket = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test else "val" if bucket < test + val else "train"


def seed_of(text: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()[:8], 16)


def src(dataset: str, config: str | None, row_id) -> dict:
    return {"dataset": dataset, "config": config, "row_id": str(row_id), "license": LICENSES[dataset]}


def target_text(row: dict) -> str:
    for message in reversed(row["messages"]):
        if message["role"] == "assistant":
            content = message.get("content") or ""
            return content if content else json.dumps(message.get("tool_calls") or [], sort_keys=True)
    return json.dumps(row.get("reference") or {}, sort_keys=True) + row["messages"][-1].get("content", "")


def hub_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.S).strip()


def render_page(pdf_bytes: bytes, page_number: int):
    """Render like Saara's runtime (PyMuPDF at RENDER_DPI), but cap the longest side at MAX_SIDE px.
    Oversized scans (LoC pages reach 5555x7072 at 150 DPI) are shrunk by Gemma's image processor to the
    1120-token budget anyway, so the cap changes cost, not what the model sees."""
    import fitz
    from PIL import Image

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        # olmOCR tarballs hold single-page extracts; page_number refers to the source document.
        page = doc[0 if len(doc) == 1 else page_number - 1]
        longest_pt = max(page.rect.width, page.rect.height)
        dpi = min(RENDER_DPI, MAX_SIDE * 72 / longest_pt)
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ---------- OCR: olmOCR-mix-1025 (+ degraded copies) ----------

def _ocr_row(rid, source, record, rel, split, group, meta) -> dict:
    return {
        "id": rid, "skill": "ocr", "split": split, "group": group,
        "source": src("allenai/olmOCR-mix-1025", source, record["id"]),
        "messages": [
            {"role": "system", "content": SYSTEM["ocr"]},
            {"role": "user", "content": [{"type": "image", "path": str(rel)}, {"type": "text", "text": OCR_USER}]},
            {"role": "assistant", "content": record["natural_text"].strip()},
        ],
        "meta": {"ocr_source": source, "is_table": record.get("is_table"), "is_diagram": record.get("is_diagram"),
                 "language": record.get("primary_language"), **meta},
    }


def build_ocr(out: Path, plan: dict, noisy_share: float) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from PIL import Image

    repo = "allenai/olmOCR-mix-1025"
    tar_names = sorted(f.path for f in HfApi().list_repo_tree(repo, path_in_repo="pdf_tarballs", repo_type="dataset"))
    def one_source(source: str, n_train: int, n_tars: int, n_test: int, aug_share: float) -> list[dict]:
      rows: list[dict] = []
      for split_name, budget in (("eval", n_test), ("train", n_train)):
            tars = [t for t in tar_names if Path(t).name.startswith(f"{source}_{split_name}_")]
            wanted = {Path(t).name for t in (tars[:n_tars] if split_name == "train" else tars)}
            by_tar: dict[str, list[dict]] = defaultdict(list)
            for record in pq.read_table(hub_file(repo, f"{source}_{split_name}.parquet")).to_pylist():
                tar_name, _, member = record["pdf_relpath"].partition(":")
                if Path(tar_name).name not in wanted:
                    continue
                if not record.get("is_rotation_valid") or record.get("rotation_correction"):
                    drops["ocr_rotation"] += 1
                    continue
                if not (record.get("natural_text") or "").strip() or len(record["natural_text"]) > 12000:
                    drops["ocr_empty_or_too_long"] += 1
                    continue
                by_tar[tar_name].append({**record, "_member": member})
            taken = 0
            for tar_name, records in sorted(by_tar.items()):
                if taken >= budget:
                    break
                random.Random(SEED).shuffle(records)
                members = {r["_member"]: r for r in records[: budget - taken]}
                with tarfile.open(hub_file(repo, tar_name), "r:gz") as archive:
                    for info in archive:
                        record = members.get(info.name)
                        if record is None:
                            continue
                        stem = record["id"].replace("/", "_")
                        rel = Path("images/ocr") / source / f"{stem}.png"
                        (out / rel).parent.mkdir(parents=True, exist_ok=True)
                        if (out / rel).exists():  # resume: reuse pages rendered by an earlier run
                            image = Image.open(out / rel).convert("RGB")
                        else:
                            try:
                                image = render_page(archive.extractfile(info).read(), int(record["page_number"]))
                            except Exception:  # corrupt PDFs are dropped, not fatal
                                drops["ocr_render_failed"] += 1
                                continue
                            image.save(out / rel, compress_level=1)
                        doc = record["id"].rsplit("-", 1)[0]
                        group = f"{source}:{doc}"
                        if split_name == "eval":
                            split = "test"
                        else:
                            split = "val" if split_of(group, 0.05, 0.0) == "val" else "train"
                        base_id = f"ocr:{source}:{record['id']}"
                        rows.append(_ocr_row(base_id, source, record, rel, split, group, {"degraded": []}))
                        share = noisy_share if split == "test" else aug_share
                        if split != "val" and random.Random(seed_of(base_id)).random() < share:
                            noisy, ops = degrade(image, seed_of(base_id + ":aug"))
                            noisy_rel = rel.with_name(f"{stem}.noisy.png")
                            noisy.save(out / noisy_rel, compress_level=1)
                            row = _ocr_row(base_id + ":noisy", source, record, noisy_rel, split, group, {"degraded": ops})
                            row["derived_from"] = base_id
                            rows.append(row)
                        taken += 1
            print(f"ocr {source}/{split_name}: {taken}")
      return rows

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(len(plan)) as pool:
        parts = list(pool.map(lambda item: one_source(item[0], *item[1]), plan.items()))
    return [row for part in parts for row in part]


# ---------- Cleaning: Post-OCR-Correction + fineweb clutter ----------

def build_post_ocr(caps: dict) -> list[dict]:
    from datasets import load_dataset

    repo, rows = "PleIAs/Post-OCR-Correction", []
    for record in load_dataset(repo, "english", split="train", streaming=True):
        if len(rows) >= caps["post_ocr"]:
            break
        noisy, clean = (record.get("text") or "").strip(), (record.get("corrected_text") or "").strip()
        if not 80 <= (record.get("word_count") or 0) <= 900 or not clean:
            drops["post_ocr_length"] += 1
            continue
        if not 0.5 <= len(clean) / max(1, len(noisy)) <= 1.5:
            drops["post_ocr_ratio"] += 1
            continue
        group = f"postocr:{record['id']}"  # newspaper title (LCCN): test = unseen publications
        rows.append({
            "id": f"clean:postocr:{record['index_id']}", "skill": "clean", "split": split_of(group), "group": group,
            "source": src(repo, "english", record["index_id"]),
            "messages": [{"role": "system", "content": SYSTEM["clean"]},
                         {"role": "user", "content": CLEAN_USER.format(text=noisy)},
                         {"role": "assistant", "content": clean}],
            "meta": {"kind": "ocr_noise", "date": record.get("date")},
        })
    print(f"post_ocr: {len(rows)}")
    return rows


def build_clutter(caps: dict) -> list[dict]:
    from datasets import load_dataset

    repo, rows = "HuggingFaceFW/fineweb-edu", []
    for record in load_dataset(repo, "sample-10BT", split="train", streaming=True):
        if len(rows) >= caps["clutter"]:
            break
        text = (record.get("text") or "").strip()
        if not 120 <= len(text.split()) <= 600:
            continue
        cluttered, kinds = add_clutter(text, seed_of(record["id"]))
        group = "domain:" + (urlparse(record.get("url") or "").netloc or record["id"])
        rows.append({
            "id": f"clean:clutter:{record['id']}", "skill": "clean", "split": split_of(group), "group": group,
            "source": src(repo, "sample-10BT", record["id"]),
            "messages": [{"role": "system", "content": SYSTEM["clean"]},
                         {"role": "user", "content": CLEAN_USER.format(text=cluttered)},
                         {"role": "assistant", "content": text}],
            "meta": {"kind": "web_clutter", "clutter": kinds},
        })
    print(f"clutter: {len(rows)}")
    return rows


# ---------- Workflows: Toucan, Hermes, When2Call ----------

def _call(name: str, args: dict, index: int) -> dict:
    return {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": args}}


def _attach_calls(messages: list[dict], calls: list[dict]) -> None:
    if not messages or messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
        messages.append({"role": "assistant", "content": ""})
    messages[-1].setdefault("tool_calls", []).extend(calls)


def _invalid(messages: list[dict], tool_names: set[str]) -> str | None:
    pending: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] not in tool_names:
                return "unknown_tool"
            if not isinstance(call["function"]["arguments"], dict):
                return "args_not_object"
            pending.append(call["id"])
        if message["role"] == "tool":
            if message.get("tool_call_id") not in pending:
                return "orphan_tool_response"
            pending.remove(message["tool_call_id"])
    if pending:
        return "unanswered_call"
    last = messages[-1]
    return None if last["role"] == "assistant" and not last.get("tool_calls") and last.get("content") else "no_final_answer"


def _reference(messages: list[dict]) -> dict | None:
    """Eval target: the first assistant turn that calls tools, predicted from everything before it."""
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and message.get("tool_calls"):
            return {"prompt_len": index, "tool_calls": [c["function"] for c in message["tool_calls"]]}
    return None


def _workflow_row(rid, dataset, config, row_id, group, messages, tools, meta=None, split=None) -> dict | None:
    reason = _invalid(messages, {t["function"]["name"] for t in tools})
    if reason:
        drops[f"workflow_{reason}"] += 1
        return None
    row = {"id": rid, "skill": "workflow", "split": split or split_of(group), "group": group,
           "source": src(dataset, config, row_id), "tools": tools, "meta": meta or {},
           "messages": [{"role": "system", "content": SYSTEM["workflow"]}, *messages]}
    if row["split"] == "test":
        row["reference"] = _reference(row["messages"])
        if row["reference"] is None:
            drops["workflow_test_without_call"] += 1
            return None
    return row


def build_toucan(caps: dict) -> list[dict]:
    import pyarrow.parquet as pq

    repo, rows = "Agent-Ark/Toucan-1.5M", []
    records = pq.read_table(hub_file(repo, "SFT/train-00000-of-00003.parquet")).to_pylist()
    random.Random(SEED).shuffle(records)
    for record in records:
        if len(rows) >= caps["toucan"]:
            break
        try:
            tools = json.loads(record["tools"]) if isinstance(record["tools"], str) else record["tools"]
            raw = json.loads(record["messages"]) if isinstance(record["messages"], str) else record["messages"]
        except (TypeError, json.JSONDecodeError):
            drops["toucan_parse"] += 1
            continue
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools or []]
        messages, pending, counter, ok = [], [], 0, bool(tools)
        for message in raw:
            role, content = message.get("role"), message.get("content") or ""
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": _strip_think(content)})
            elif role == "tool_call":
                try:
                    call = parse_python_literal(content) if isinstance(content, str) else content
                    args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
                except (ValueError, SyntaxError, KeyError, TypeError, json.JSONDecodeError):
                    ok = False
                    break
                counter += 1
                _attach_calls(messages, [_call(call["name"], args, counter)])
                pending.append(f"call_{counter}")
            elif role == "tool_response":
                if not pending:
                    ok = False
                    break
                messages.append({"role": "tool", "tool_call_id": pending.pop(0), "content": content[:4000]})
        if not ok:
            drops["toucan_parse"] += 1
            continue
        servers = sorted({t["function"]["name"].rsplit("-", 1)[0] for t in tools})
        row = _workflow_row(f"workflow:toucan:{record['uuid']}", repo, "SFT", record["uuid"], "mcp:" + "|".join(servers),
                            messages, tools, {"subset": record.get("subset_name"), "target_tools": record.get("target_tools")})
        if row:
            rows.append(row)
    print(f"toucan: {len(rows)}")
    return rows


def build_hermes(caps: dict) -> list[dict]:
    from datasets import load_dataset

    repo, rows = "NousResearch/hermes-function-calling-v1", []
    for record in load_dataset(repo, "func_calling", split="train"):
        if len(rows) >= caps["hermes"]:
            break
        conv = record["conversations"]
        # The system prompt mentions "<tools> </tools>" in prose before the real block: use the last non-empty match.
        blocks = re.findall(r"<tools>\s*(.*?)\s*</tools>", conv[0]["value"], re.S) if conv and conv[0]["from"] == "system" else []
        blocks = [b for b in blocks if b.strip()]
        try:
            tools = json.loads(blocks[-1]) if blocks else []
        except json.JSONDecodeError:
            tools = []
        messages, pending, counter, ok = [], [], 0, bool(tools)
        for turn in conv[1:] if ok else []:
            value = turn["value"]
            if turn["from"] == "human":
                messages.append({"role": "user", "content": value})
            elif turn["from"] == "gpt":
                messages.append({"role": "assistant", "content": re.sub(r"<tool_call>.*?</tool_call>", "", value, flags=re.S).strip()})
                for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", value, re.S):
                    try:
                        call = json.loads(block)
                    except json.JSONDecodeError:
                        ok = False
                        break
                    counter += 1
                    messages[-1].setdefault("tool_calls", []).append(_call(call["name"], call.get("arguments") or {}, counter))
                    pending.append(f"call_{counter}")
            elif turn["from"] == "tool":
                for block in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", value, re.S):
                    if not pending:
                        ok = False
                        break
                    messages.append({"role": "tool", "tool_call_id": pending.pop(0), "content": block[:4000]})
        if not ok:
            drops["hermes_parse"] += 1
            continue
        row = _workflow_row(f"workflow:hermes:{record['id']}", repo, "func_calling", record["id"],
                            f"hermes:{record['category']}", messages, tools, {"category": record["category"]})
        if row:
            rows.append(row)
    print(f"hermes: {len(rows)}")
    return rows


_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object", "list": "array"}


def _norm_tool(raw) -> dict:
    spec = json.loads(raw) if isinstance(raw, str) else raw
    params = spec.get("parameters") or {}
    props = {}
    for name, prop in (params.get("properties") or {}).items():
        kind = str(prop.get("type", "string")).split(",")[0].strip()
        props[name] = {"type": _TYPES.get(kind, kind), "description": prop.get("description", "")}
    required = params.get("required") or spec.get("required") or []
    return {"type": "function", "function": {"name": spec["name"], "description": spec.get("description", ""),
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


def build_when2call(caps: dict) -> list[dict]:
    from datasets import load_dataset

    repo, rows = "nvidia/When2Call", []
    for index, record in enumerate(load_dataset(repo, "train_sft", split="train")):
        if len(rows) >= caps["when2call"]:
            break
        try:
            tools = [_norm_tool(t) for t in record["tools"]]
        except (KeyError, TypeError, json.JSONDecodeError):
            drops["when2call_tools"] += 1
            continue
        group = "w2c:" + "|".join(sorted(t["function"]["name"] for t in tools))
        split = "val" if split_of(group, 0.05, 0.0) == "val" else "train"  # official MCQ set is the test
        row = _workflow_row(f"workflow:when2call:{index}", repo, "train_sft", index, group, record["messages"], tools,
                            {"kind": "restraint"}, split=split)
        if row:
            rows.append(row)
    tests = 0
    for record in load_dataset(repo, "test", split="mcq"):
        if tests >= caps["when2call_test"]:
            break
        try:
            tools = [_norm_tool(t) for t in record.get("tools") or []]
        except (KeyError, TypeError, json.JSONDecodeError):
            drops["when2call_test_tools"] += 1
            continue
        rows.append({
            "id": f"workflow:when2call_test:{record['uuid']}", "skill": "workflow", "split": "test", "group": "w2c_test",
            "source": src(repo, "test/mcq", record["uuid"]), "tools": tools,
            "messages": [{"role": "system", "content": SYSTEM["workflow"]}, {"role": "user", "content": record["question"]}],
            "reference": {"prompt_len": 2, "when2call": record["correct_answer"]},
        })
        tests += 1
    print(f"when2call: {len(rows)} (test {tests})")
    return rows


# ---------- Replay: general instruction data against forgetting ----------

def build_replay(caps: dict) -> list[dict]:
    from datasets import load_dataset

    repo, rows = "allenai/tulu-3-sft-mixture", []
    for record in load_dataset(repo, split="train", streaming=True):
        if len(rows) >= caps["replay"]:
            break
        messages = [{"role": m["role"], "content": m["content"]} for m in record["messages"] if m["role"] in ("user", "assistant")]
        if len(messages) < 2 or messages[-1]["role"] != "assistant" or sum(len(m["content"]) for m in messages) > 4000:
            continue
        split = "val" if split_of(record["id"], 0.05, 0.0) == "val" else "train"
        rows.append({"id": f"general:tulu:{record['id']}", "skill": "general", "split": split, "group": f"tulu:{record['id']}",
                     "source": {**src(repo, None, record["id"]), "subset": record.get("source")},
                     "messages": [{"role": "system", "content": SYSTEM["general"]}, *messages]})
    print(f"replay: {len(rows)}")
    return rows


# ---------- dedup / contamination ----------

def dedup(rows: list[dict]) -> list[dict]:
    """Exact + MinHash near-duplicate removal on targets. Test rows claim first, so train/val
    rows that duplicate a test row are dropped (contamination guard). Degraded copies follow their base row."""
    from datasketch import MinHash, MinHashLSH

    def minhash(text: str) -> MinHash:
        mh = MinHash(num_perm=64, seed=SEED)
        words = re.findall(r"\w+", text.lower())
        for shingle in {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}:
            mh.update(shingle.encode())
        return mh

    base = [r for r in rows if "derived_from" not in r]
    derived = [r for r in rows if "derived_from" in r]
    lsh, kept, exact = MinHashLSH(threshold=0.8, num_perm=64), [], set()
    for row in sorted(base, key=lambda r: {"test": 0, "val": 1, "train": 2}[r["split"]]):
        text = target_text(row)
        digest = hashlib.sha256((row["skill"] + text).encode()).hexdigest()
        if digest in exact:
            drops[f"dedup_exact_{row['split']}"] += 1
            continue
        mh = minhash(text) if len(text) > 200 else None
        if mh is not None and lsh.query(mh):
            drops[f"dedup_near_{row['split']}"] += 1
            continue
        exact.add(digest)
        if mh is not None:
            lsh.insert(row["id"], mh)
        kept.append(row)
    kept_ids = {r["id"] for r in kept}
    kept.extend(r for r in derived if r["derived_from"] in kept_ids)
    return kept


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes for an end-to-end check")
    parser.add_argument("--caps", type=json.loads, default={}, help='override caps, e.g. \'{"toucan": 8000}\'')
    parser.add_argument("--skip", nargs="*", default=[], help="ocr post_ocr clutter toucan hermes when2call replay")
    args = parser.parse_args()
    caps = {**(SMOKE_CAPS if args.smoke else DEFAULT_CAPS), **args.caps}
    plan = {s: (4, 1, 3, 0.5) for s in ("00_documents", "02_loc_transcripts")} if args.smoke else OCR_PLAN
    args.out.mkdir(parents=True, exist_ok=True)

    builders = {"ocr": lambda: build_ocr(args.out, plan, caps["noisy_test_share"]), "post_ocr": lambda: build_post_ocr(caps),
                "clutter": lambda: build_clutter(caps), "toucan": lambda: build_toucan(caps),
                "hermes": lambda: build_hermes(caps), "when2call": lambda: build_when2call(caps),
                "replay": lambda: build_replay(caps)}
    from concurrent.futures import ThreadPoolExecutor

    active = [build for name, build in builders.items() if name not in args.skip]
    with ThreadPoolExecutor(len(active)) as pool:
        rows = [row for part in pool.map(lambda build: build(), active) for row in part]
    rows.sort(key=lambda r: r["id"])  # deterministic order regardless of thread timing
    before = len(rows)
    rows = dedup(rows)
    print(f"dedup: {before} -> {len(rows)}")

    from huggingface_hub import HfApi

    api, files = HfApi(), {}
    for split in ("train", "val", "test"):
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row["split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": sum(r["split"] == split for r in rows)}
    counts = Counter((r["split"], r["skill"]) for r in rows)
    manifest = {
        "prompt_version": PROMPT_VERSION, "seed": SEED, "caps": caps, "ocr_plan": plan,
        "datasets": {repo: {"license": lic, "revision": api.dataset_info(repo).sha} for repo, lic in LICENSES.items()},
        "files": files, "counts": {f"{s}/{k}": n for (s, k), n in sorted(counts.items())},
        "test_row_ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in rows if r["split"] == "test")).encode()).hexdigest(),
        "drops": dict(drops),
    }
    (args.out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("files", "counts", "drops")}, indent=2))


if __name__ == "__main__":
    main()
