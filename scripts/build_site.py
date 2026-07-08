"""Build the GitHub Pages site (docs/) from the canonical markdown docs.

Renders methodology.md and dataset/DATASET_CARD.md into themed HTML pages that
share a nav + stylesheet with the leaderboard. The markdown files stay the
single source of truth; run this after editing them:

    python scripts/build_site.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
GITHUB = "https://github.com/yourbuddyconner/epstein-bench"

# -- minimal markdown -> HTML (covers what our docs use) -----------------------


def _inline(text: str) -> str:
    text = html.escape(text)
    # protect inline code spans from further substitution
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)
    return text


def _render_table(rows: list[str]) -> str:
    def cells(line: str) -> list[str]:
        # split on unescaped pipes, then unescape \| within a cell
        parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        return [p.strip().replace("\\|", "|") for p in parts]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    out = ['<div class="tablewrap"><table><thead><tr>']
    out += [f"<th>{_inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):  # code fence
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
        elif line.lstrip().startswith("|") and i + 1 < n and set(lines[i + 1].strip()) <= set("|-: "):
            tbl = []
            while i < n and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.append(_render_table(tbl))
        elif re.match(r"^#{1,4}\s", line):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{_inline(line.lstrip('#').strip())}</h{level}>")
            i += 1
        elif line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(_inline(lines[i].lstrip(">").strip()))
                i += 1
            out.append("<blockquote>" + " ".join(buf) + "</blockquote>")
        elif re.match(r"^\s*[-*]\s", line):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s", lines[i]):
                buf.append("<li>" + _inline(re.sub(r"^\s*[-*]\s", "", lines[i])) + "</li>")
                i += 1
            out.append("<ul>" + "".join(buf) + "</ul>")
        elif re.match(r"^\s*\d+\.\s", line):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s", lines[i]):
                buf.append("<li>" + _inline(re.sub(r"^\s*\d+\.\s", "", lines[i])) + "</li>")
                i += 1
            out.append("<ol>" + "".join(buf) + "</ol>")
        elif set(line.strip()) == {"-"}:
            out.append("<hr>")
            i += 1
        else:
            # always consume the current line so i advances (a stray '|' line
            # that isn't a valid table must not spin here forever)
            buf = [_inline(line.strip())]
            i += 1
            while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|>|\s*[-*]\s|\s*\d+\.\s|\||```)", lines[i]):
                buf.append(_inline(lines[i].strip()))
                i += 1
            out.append("<p>" + " ".join(buf) + "</p>")
    return "\n".join(out)


# -- shared shell --------------------------------------------------------------

STYLE = """
  :root { --bg:#faf9f6; --fg:#1a1a1a; --muted:#6b6b6b; --line:#e2ddd4;
          --accent:#8b2c2c; --card:#ffffff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16150f; --fg:#ece9e2; --muted:#9a968c; --line:#33312a;
            --accent:#d47b6a; --card:#1e1d16; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.65 Georgia,'Times New Roman',serif; }
  nav { border-bottom:1px solid var(--line); position:sticky; top:0;
        background:color-mix(in srgb, var(--bg) 92%, transparent);
        backdrop-filter:blur(6px); z-index:10; }
  nav .inner { max-width:60rem; margin:0 auto; padding:.9rem 1.25rem;
        display:flex; gap:1.4rem; align-items:baseline; flex-wrap:wrap; }
  nav .brand { font-weight:700; letter-spacing:-.01em; margin-right:.4rem; }
  nav a { color:var(--fg); text-decoration:none; font-size:.95rem; }
  nav a.active { color:var(--accent); }
  nav a:hover { color:var(--accent); }
  nav .spacer { flex:1; }
  main { max-width:60rem; margin:0 auto; padding:2.5rem 1.25rem 5rem; }
  h1 { font-size:2.2rem; margin:0 0 .3rem; letter-spacing:-.01em; }
  h2 { font-size:1.35rem; margin:2.4rem 0 .7rem; }
  h3 { font-size:1.08rem; margin:1.8rem 0 .5rem; }
  h4 { font-size:.98rem; margin:1.4rem 0 .4rem; color:var(--muted); }
  .sub { color:var(--muted); margin:0 0 2rem; font-size:1.05rem; }
  a { color:var(--accent); }
  p, li { max-width:44rem; }
  ul, ol { padding-left:1.4rem; }
  li { margin:.25rem 0; }
  blockquote { margin:1rem 0; padding:.4rem 0 .4rem 1rem;
        border-left:3px solid var(--line); color:var(--muted); }
  hr { border:none; border-top:1px solid var(--line); margin:2rem 0; }
  .tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px;
        background:var(--card); margin:1rem 0; }
  table { border-collapse:collapse; width:100%;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }
  th, td { padding:.55rem .8rem; text-align:left; border-bottom:1px solid var(--line);
        vertical-align:top; }
  th { color:var(--muted); font-weight:600; white-space:nowrap; }
  tr:last-child td { border-bottom:none; }
  code { font-size:.85em; background:var(--card); border:1px solid var(--line);
        padding:.08em .35em; border-radius:4px; }
  pre { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:.9rem 1rem; overflow-x:auto; }
  pre code { background:none; border:none; padding:0; font-size:.82rem; }
  .note { color:var(--muted); font-size:.9rem; }
  td.headline { font-weight:700; }
  .empty { padding:2.5rem 1rem; text-align:center; color:var(--muted); font-style:italic; }
  .lede { font-size:1.25rem; line-height:1.5; max-width:40rem; margin:0 0 1.5rem; }
  .kicker { text-transform:uppercase; letter-spacing:.12em; font-size:.72rem;
        font-weight:700; color:var(--accent); font-family:ui-monospace,Menlo,monospace; }
  .files { display:grid; gap:.75rem; margin:1.2rem 0 1.8rem; }
  .file { border:1px solid var(--line); border-left:3px solid var(--accent);
        border-radius:6px; background:var(--card); padding:.85rem 1rem; }
  .file .q { font-weight:600; }
  .file .a { color:var(--muted); font-size:.92rem; margin-top:.3rem; }
  .file .a b { color:var(--fg); font-style:italic; font-weight:400; }
  .pull { font-size:1.5rem; line-height:1.4; margin:1.6rem 0; padding-left:1.1rem;
        border-left:3px solid var(--accent); max-width:40rem; }
  .pull cite { display:block; font-size:.85rem; color:var(--muted); font-style:normal;
        margin-top:.5rem; font-family:ui-monospace,Menlo,monospace; }
  .disclaimer { border:1px solid var(--line); border-radius:8px; background:var(--card);
        padding:1rem 1.2rem; margin:2.5rem 0 0; color:var(--muted); font-size:.9rem; }
"""


def nav(active: str) -> str:
    def link(href: str, label: str, key: str) -> str:
        cls = ' class="active"' if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    return (
        '<nav><div class="inner">'
        '<span class="brand">Epstein Bench</span>'
        + link("index.html", "Leaderboard", "home")
        + link("methodology.html", "Methodology", "methodology")
        + link("dataset.html", "Dataset Card", "dataset")
        + '<span class="spacer"></span>'
        + f'<a href="{GITHUB}">GitHub</a>'
        "</div></nav>"
    )


def page(title: str, active: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{STYLE}</style>
</head>
<body>
{nav(active)}
<main>
{body}
</main>
</body>
</html>
"""


# -- leaderboard page (has live JS) --------------------------------------------

LEADERBOARD_BODY = """
  <h1>Epstein Bench</h1>
  <p class="lede">Can your AI find the one sentence that matters in more than
  80,000 scanned, garbled, redaction-strewn government documents?</p>

  <p>Every other retrieval benchmark quizzes AI on clean Wikipedia. The real
  world looks nothing like that: OCR wreckage, near-duplicate emails, endless
  legalese, and the one fact you need buried on page 4,000. <strong>Epstein Bench
  is that world.</strong> It's built from the full public Epstein Files: about
  80,000 real documents released by courts and Congress, turned into 1,000
  questions a system can only answer by finding the right page and citing it.</p>

  <p class="kicker">From the files</p>
  <p>Real, verified questions in the benchmark. Each answer is a single sentence
  hiding somewhere in the corpus:</p>
  <div class="files">
    <div class="file">
      <div class="q">Who did Jeffrey Epstein ask to find him "the best codebreaker, NSA type"?</div>
      <div class="a">Buried in an email to a veteran TV journalist. The system has to surface the exact thread. <b>Epstein: "Can you find me the best codebreaker nsa type."</b></div>
    </div>
    <div class="file">
      <div class="q">What did Steve Bannon email Epstein about "real power"?</div>
      <div class="a"><b>"we r on the cusp of real power."</b> One line in a chain of emails between Epstein and the former White House strategist.</div>
    </div>
    <div class="file">
      <div class="q">Who was on the guest list for Epstein's dinner on September 20, 2013?</div>
      <div class="a">A calendar entry: <b>"DINNER W/ BILL GATES, TERJE, JAGBLAND, OTHERS."</b> The corpus is full of these. The trick is retrieving the right one.</div>
    </div>
    <div class="file">
      <div class="q">Which account was tied to Ghislaine Maxwell at J.P. Morgan?</div>
      <div class="a">The answer is a line item on a scanned bank statement, the kind of needle dense-vector search routinely misses on noisy OCR.</div>
    </div>
  </div>

  <p class="pull">The hardest questions ask a system to reconstruct the entire
  documented timeline of one person's contacts with Epstein, scattered across
  dozens of files. Our best baseline scores near zero.
  <cite>the "dossier" task family</cite></p>

  <h2>How it's scored, and why it's honest</h2>
  <p>A system earns a point only when it gets the answer right <strong>and</strong>
  cites the document that proves it. No confident bluffing: the thing language
  models do worst is rewarded least. We also run a control that gets no documents
  at all, only the question. It scores <strong>zero</strong> on every fact, which
  proves these answers can't be guessed. That same control doubles as a
  <strong>training-contamination probe</strong>: a measure of how much of the
  Epstein Files a model already absorbed from its training data, a number that
  will only climb as new models are trained on this public release.</p>
  <p class="note">Full details in the <a href="methodology.html">methodology</a>
  and <a href="dataset.html">dataset card</a>. Every task survived a four-stage
  verification gauntlet before release.</p>

  <h2>Leaderboard: <code>full</code> split, dataset v1.0</h2>
  <div class="tablewrap">
    <table id="board">
      <thead><tr>
        <th>system</th><th>overall</th><th>uncited</th>
        <th>single_hop</th><th>aggregation</th><th>timeline</th><th>dossier</th>
        <th>unanswerable</th><th>recall@5</th><th>recall@20</th><th>ndcg@10</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    <div class="empty" id="empty" hidden>No verified submissions yet.</div>
  </div>
  <p class="note">Scores are recomputed by CI from raw predictions; self-reported
  numbers are never used. <strong>uncited</strong> = correctness ignoring the
  citation gate (for the <code>parametric</code> baseline, a probe of how much of
  the corpus a model already knows from training). Submit via PR. See the
  <a href="%GITHUB%#submitting-to-the-leaderboard">README</a>.</p>

  <div class="disclaimer">These are public records released by U.S. courts and
  Congress. Appearing in the files means appearing in someone's email, calendar,
  or financial records. It is not an accusation of wrongdoing. Epstein Bench
  measures whether AI can retrieve and cite what the documents say; it takes no
  position on anyone's conduct.</div>

<script>
  const fmt = x => (typeof x === "number" ? x.toFixed(3) : "n/a");
  fetch("leaderboard.json").then(r => r.json()).then(data => {
    const rows = data.entries || [];
    if (!rows.length) { document.getElementById("empty").hidden = false; return; }
    const tb = document.querySelector("#board tbody");
    for (const e of rows) {
      const t = e.per_type || {}, r = e.retrieval || {};
      const cells = [e.system_name || "?", fmt(e.overall_cited_correctness),
        fmt(e.overall_uncited_correctness), fmt(t.single_hop), fmt(t.aggregation),
        fmt(t.timeline), fmt(t.dossier), fmt(t.unanswerable),
        fmt(r["recall@5"]), fmt(r["recall@20"]), fmt(r["ndcg@10"])];
      const tr = document.createElement("tr");
      cells.forEach((v, i) => { const td = document.createElement("td");
        if (i === 1) td.className = "headline"; td.textContent = v; tr.appendChild(td); });
      tb.appendChild(tr);
    }
  }).catch(() => { document.getElementById("empty").hidden = false; });
</script>
""".replace("%GITHUB%", GITHUB)


def main() -> None:
    (DOCS / "index.html").write_text(page("Epstein Bench: Leaderboard", "home", LEADERBOARD_BODY))

    methodology = (DOCS / "methodology.md").read_text()
    (DOCS / "methodology.html").write_text(
        page("Epstein Bench: Methodology", "methodology", md_to_html(methodology))
    )

    card = (REPO / "dataset" / "DATASET_CARD.md").read_text()
    (DOCS / "dataset.html").write_text(
        page("Epstein Bench: Dataset Card", "dataset", md_to_html(card))
    )
    print("wrote docs/index.html, docs/methodology.html, docs/dataset.html")


if __name__ == "__main__":
    main()
