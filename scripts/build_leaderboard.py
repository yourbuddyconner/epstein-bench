"""Rebuild docs/leaderboard.json from validated submission bundles.

Also bakes the leaderboard tables and the hero stat into docs/index.html
(between ``generated:*`` comment markers), so the page needs no JavaScript.

Run after merging a submission PR (CI verified the committed scores.json
against a recomputation):
    python scripts/build_leaderboard.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"
OUT = REPO / "docs" / "leaderboard.json"
INDEX = REPO / "docs" / "index.html"

# Human-readable names for the reference systems; unknown system_names pass
# through unchanged, so third-party submissions keep whatever name they chose.
DISPLAY_NAMES = {
    "agentic-sonnet-5 (reference)": "Claude Sonnet 5 agent",
    "agentic-opus-4-8 (reference)": "Claude Opus 4.8 agent",
    "hybrid (reference)": "Hybrid retrieval (BM25 + dense)",
    "bm25 (reference)": "BM25 keyword search",
    "dense (reference)": "Dense embeddings",
    "closed_book (reference)": "Closed book",
    "parametric (reference)": "Parametric probe",
}
CONTROLS = {"closed_book (reference)", "parametric (reference)"}
REFERENCE_SUFFIX = " (reference)"

REQUIRED = (
    "system_name",
    "split",
    "dataset_version",
    "overall_cited_correctness",
    "overall_cited_correctness_ci95",
    "per_type",
    "n_tasks",
    "judge_model",
    "judge_prompt_version",
)


def _fmt(x, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "n/a"


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _display_name(entry: dict) -> tuple[str, str]:
    """Return (display name, tag). Tag is 'control', 'self-run', or ''."""
    raw = entry["system_name"]
    name = DISPLAY_NAMES.get(raw, raw)
    if raw in CONTROLS:
        return name, "control"
    if raw.endswith(REFERENCE_SUFFIX):
        return name, "self-run"
    return name, ""


def _hero_html(entries: list[dict]) -> str:
    top = next(e for e in entries if e["system_name"] not in CONTROLS)
    name, _ = _display_name(top)
    pct = round(top["overall_cited_correctness"] * 100)
    cost = top.get("cost_usd_per_task")
    cost_line = (
        f", at about ${cost:.2f} a question" if isinstance(cost, (int, float)) else ""
    )
    return (
        '  <div class="hero">\n'
        f'    <div class="big">{pct}%</div>\n'
        '    <div class="caption">\n'
        f"      <p>The best system on the board, {html.escape(name)}, answers\n"
        f"      {pct}% of the questions correctly with a valid citation{cost_line}.</p>\n"
        '      <p class="cta"><a href="https://github.com/yourbuddyconner/epstein-bench'
        '#submitting-to-the-leaderboard">Think yours does better? Submit a run.</a></p>\n'
        "    </div>\n"
        "  </div>"
    )


def _board_rows(entries: list[dict]) -> str:
    rows = []
    rank = 0
    for e in entries:
        name, tag = _display_name(e)
        is_control = tag == "control"
        if not is_control:
            rank += 1
        lo, hi = e["overall_cited_correctness_ci95"]
        tag_html = f'<span class="tag">{tag}</span>' if tag else ""
        cost = e.get("cost_usd_per_task")
        cost_cell = f"${cost:.3f}" if isinstance(cost, (int, float)) else "n/a"
        row_class = ' class="control"' if is_control else ""
        rank_cell = "&mdash;" if is_control else str(rank)
        rows.append(
            f"      <tr{row_class}>"
            f"<td>{rank_cell}</td>"
            f"<td>{html.escape(name)}{tag_html}</td>"
            f'<td class="headline">{_pct(e["overall_cited_correctness"])}</td>'
            f'<td class="ci">{lo * 100:.1f} to {hi * 100:.1f}</td>'
            f'<td class="cost">{cost_cell}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _detail_rows(entries: list[dict]) -> str:
    rows = []
    for e in entries:
        t = e.get("per_type") or {}
        r = e.get("retrieval") or {}
        tok = e.get("tokens_per_task")
        cells = [
            html.escape(e["system_name"]),
            _fmt(e["overall_cited_correctness"]),
            _fmt(e.get("overall_cited_correctness_micro")),
            _fmt(e.get("overall_uncited_correctness")),
            _fmt(t.get("single_hop")),
            _fmt(t.get("aggregation")),
            _fmt(t.get("timeline")),
            _fmt(t.get("dossier")),
            _fmt(t.get("unanswerable")),
            _fmt(t.get("false_premise")),
            _fmt(e.get("citation_precision")),
            _fmt(r.get("recall@5")),
            _fmt(r.get("recall@20")),
            f"{tok:,.0f}" if isinstance(tok, (int, float)) else "n/a",
        ]
        rows.append("        <tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "\n".join(rows)


def _inject(page: str, marker: str, content: str) -> str:
    pattern = re.compile(
        rf"(<!-- generated:{marker}:start -->).*?(<!-- generated:{marker}:end -->)",
        re.DOTALL,
    )
    if not pattern.search(page):
        raise SystemExit(f"docs/index.html: missing generated:{marker} markers")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n  {m.group(2)}", page, count=1)


def render_index(entries: list[dict]) -> None:
    page = INDEX.read_text()
    page = _inject(page, "hero", _hero_html(entries))
    page = _inject(page, "board", _board_rows(entries))
    page = _inject(page, "detail", _detail_rows(entries))
    INDEX.write_text(page)
    print(f"baked {len(entries)} rows into {INDEX}")


def main() -> None:
    entries = []
    seen_names: set[str] = set()
    if SUBMISSIONS.exists():
        for scores_path in sorted(SUBMISSIONS.glob("*/scores.json")):
            report = json.loads(scores_path.read_text())
            bundle = scores_path.parent
            missing = [k for k in REQUIRED if k not in report]
            if missing:
                raise SystemExit(f"{scores_path}: missing fields {missing}")
            meta = json.loads((bundle / "metadata.json").read_text())
            for key in ("system_name", "split", "dataset_version"):
                if meta.get(key) != report.get(key):
                    raise SystemExit(
                        f"{bundle.name}: metadata.json {key}={meta.get(key)!r} "
                        f"!= scores.json {report.get(key)!r}"
                    )
            if report["split"] != "full":
                continue  # dev runs are not leaderboard-eligible
            if report["system_name"] in seen_names:
                raise SystemExit(f"duplicate system_name {report['system_name']!r}")
            seen_names.add(report["system_name"])
            entries.append(report)
    entries.sort(key=lambda r: -r["overall_cited_correctness"])
    OUT.write_text(json.dumps({"entries": entries}, indent=2))
    print(f"wrote {OUT} with {len(entries)} entries")
    render_index(entries)


if __name__ == "__main__":
    main()
