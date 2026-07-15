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
BASE_URL = "https://epsteinbench.com"
OG_IMAGE = f"{BASE_URL}/og-image.png"  # 1200x630 link-preview card (docs/og-image.png)

# -- minimal markdown -> HTML (covers what our docs use) -----------------------


def _inline(text: str) -> str:
    text = html.escape(text)
    # protect inline code spans from further substitution
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    # footnote reference: [^3] -> superscript link to the note
    text = re.sub(
        r"\[\^(\w+)\]",
        r'<sup class="fnref" id="fnref-\1"><a href="#fn-\1">\1</a></sup>',
        text,
    )
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*((?:[^*]|\*(?!\*))+)\*\*", r"<strong>\1</strong>", text)
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


def _dedupe_fnref_ids(html_text: str) -> str:
    """A note cited twice would emit duplicate id="fnref-N"; keep the first."""
    seen: set[str] = set()

    def strip_dupe(m: re.Match) -> str:
        if m.group(1) in seen:
            return '<sup class="fnref">'
        seen.add(m.group(1))
        return m.group(0)

    return re.sub(r'<sup class="fnref" id="fnref-(\w+)">', strip_dupe, html_text)


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
        elif re.match(r"^\[\^\w+\]:", line):
            # footnote definitions: consecutive [^N]: lines (with indented
            # continuations) render in place as the numbered notes list
            items = []
            while i < n and (m := re.match(r"^\[\^(\w+)\]:\s*(.*)", lines[i])):
                num, buf = m.group(1), [m.group(2)]
                i += 1
                while i < n and lines[i].startswith("  ") and lines[i].strip():
                    buf.append(lines[i].strip())
                    i += 1
                items.append(
                    f'<li id="fn-{num}">{_inline(" ".join(buf))} '
                    f'<a class="fnback" href="#fnref-{num}" '
                    f'aria-label="back to reference {num}">&#8617;</a></li>'
                )
                while i < n and not lines[i].strip():
                    i += 1
            out.append('<ol class="footnotes">' + "".join(items) + "</ol>")
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
                buf.append(lines[i].lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(buf)) + "</blockquote>")
        elif re.match(r"^\s*[-*]\s", line) or re.match(r"^\s*\d+\.\s", line):
            # list items may wrap: lines that follow an item and don't start a
            # new block belong to the current item, not a new paragraph
            marker = r"^\s*\d+\.\s" if re.match(r"^\s*\d+\.\s", line) else r"^\s*[-*]\s"
            other_block = r"^(#{1,4}\s|>|\||```|\[\^\w+\]:)"
            items = []
            while i < n and re.match(marker, lines[i]):
                buf = [re.sub(marker, "", lines[i]).strip()]
                i += 1
                while (
                    i < n
                    and lines[i].strip()
                    and not re.match(marker, lines[i])
                    and not re.match(other_block, lines[i])
                ):
                    buf.append(lines[i].strip())
                    i += 1
                items.append("<li>" + _inline(" ".join(buf)) + "</li>")
                while i < n and not lines[i].strip() and i + 1 < n and re.match(marker, lines[i + 1]):
                    i += 1
            tag = "ol" if marker == r"^\s*\d+\.\s" else "ul"
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
        elif set(line.strip()) == {"-"}:
            out.append("<hr>")
            i += 1
        else:
            # always consume the current line so i advances (a stray '|' line
            # that isn't a valid table must not spin here forever)
            buf = [line.strip()]
            i += 1
            while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|>|\s*[-*]\s|\s*\d+\.\s|\||```|\[\^\w+\]:)", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            out.append("<p>" + _inline(" ".join(buf)) + "</p>")
    return _dedupe_fnref_ids("\n".join(out))


# -- shared shell --------------------------------------------------------------

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@'
    "0,400;0,700;1,400&family=Libre+Franklin:wght@400;600;700;900&display=swap\" "
    'rel="stylesheet">'
)

STYLE = """
  :root { --desk:#d6d3ca; --sheet:#fbfaf7; --ink:#1b1a17; --muted:#6e6a61;
          --line:#c7c3b8; --sheetline:#e7e4db; --stamp:#a8231c;
          --shadow:0 1px 2px rgba(27,26,23,.14), 0 6px 18px rgba(27,26,23,.08);
          --mono:'Courier Prime',ui-monospace,Menlo,monospace;
          --sans:'Libre Franklin',system-ui,'Helvetica Neue',Arial,sans-serif;
          --serif:Georgia,'Times New Roman',serif; }
  @media (prefers-color-scheme: dark) {
    :root { --desk:#161511; --sheet:#221f19; --ink:#e8e4d9; --muted:#98927f;
            --line:#33302a; --sheetline:#37332b; --stamp:#d05a4e;
            --shadow:0 1px 2px rgba(0,0,0,.5), 0 8px 22px rgba(0,0,0,.35); } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--desk); color:var(--ink);
         font:16px/1.68 var(--serif); }
  nav { border-bottom:1px solid var(--line); position:sticky; top:0;
        background:color-mix(in srgb, var(--desk) 90%, transparent);
        backdrop-filter:blur(6px); z-index:10; font-family:var(--sans); }
  nav .inner { max-width:60rem; margin:0 auto; padding:.85rem 1.25rem;
        display:flex; gap:1.5rem; align-items:baseline; flex-wrap:wrap; }
  nav .brand { font-weight:900; letter-spacing:.06em; text-transform:uppercase;
        font-size:.9rem; margin-right:.4rem; }
  nav a { color:var(--ink); text-decoration:none; font-size:.85rem; font-weight:600; }
  nav a.active, nav a:hover { color:var(--stamp); }
  nav .spacer { flex:1; }
  main { max-width:60rem; margin:0 auto; padding:3rem 1.25rem 5rem; }
  h1 { font-family:var(--sans); font-weight:900; font-size:clamp(2rem,5vw,2.8rem);
       line-height:1.05; letter-spacing:-.02em; text-transform:uppercase;
       margin:0 0 1.2rem; }
  h2 { font-family:var(--sans); font-weight:800; font-size:1.1rem;
       text-transform:uppercase; letter-spacing:.05em; margin:2.8rem 0 .7rem; }
  h3 { font-family:var(--sans); font-weight:700; font-size:1rem;
       margin:1.8rem 0 .5rem; }
  h4 { font-family:var(--mono); font-weight:700; font-size:.78rem;
       text-transform:uppercase; letter-spacing:.12em; color:var(--muted);
       margin:1.4rem 0 .4rem; }
  a { color:var(--stamp); }
  p, li { max-width:44rem; }
  ul, ol { padding-left:1.4rem; }
  li { margin:.25rem 0; }
  blockquote { margin:1rem 0; padding:.4rem 0 .4rem 1rem;
        border-left:3px solid var(--stamp); color:var(--muted); }
  hr { border:none; border-top:1px solid var(--line); margin:2rem 0; }
  .tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:2px;
        background:var(--sheet); box-shadow:var(--shadow); margin:1rem 0;
        width:fit-content; max-width:100%; }
  table { border-collapse:collapse; width:auto; font-size:.9rem; }
  th, td { padding:.55rem 1.1rem .55rem .8rem; text-align:left;
        border-bottom:1px solid var(--sheetline); vertical-align:top; }
  td { max-width:34rem; }
  th { color:var(--muted); font-weight:700; white-space:nowrap; font-size:.7rem;
        text-transform:uppercase; letter-spacing:.08em; font-family:var(--sans); }
  tr:last-child td { border-bottom:none; }
  code { font-family:var(--mono); font-size:.85em; background:var(--sheet);
        border:1px solid var(--line); padding:.05em .35em; border-radius:2px; }
  pre { background:var(--sheet); border:1px solid var(--line); border-radius:2px;
        box-shadow:var(--shadow); padding:.9rem 1rem; overflow-x:auto; }
  pre code { background:none; border:none; padding:0; font-size:.82rem; }
  .note { color:var(--muted); font-size:.9rem; }
  sup.fnref { font-family:var(--sans); font-weight:700; line-height:0; }
  sup.fnref a { text-decoration:none; padding:0 .1em; }
  ol.footnotes { border-top:3px solid var(--stamp); padding-top:1rem;
        margin-top:1rem; font-size:.88rem; color:var(--muted); }
  ol.footnotes li { margin:.5rem 0; overflow-wrap:break-word; }
  ol.footnotes a.fnback { text-decoration:none; margin-left:.3em;
        font-family:var(--sans); }
"""


def nav(active: str) -> str:
    def link(href: str, label: str, key: str) -> str:
        cls = ' class="active"' if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    return (
        '<nav><div class="inner">'
        '<span class="brand">Epstein Bench</span>'
        + link("index.html", "Leaderboard", "home")
        + link("board.html", "The Board", "board")
        + link("methodology.html", "Methodology", "methodology")
        + link("dataset.html", "Dataset Card", "dataset")
        + '<span class="spacer"></span>'
        + f'<a href="{GITHUB}">GitHub</a>'
        "</div></nav>"
    )


def _social_meta(title: str, description: str, path: str) -> str:
    """Open Graph + Twitter Card tags so links unfurl a preview in Twitter,
    iMessage, Slack, etc. Image and URLs are absolute (required by scrapers)."""
    url = f"{BASE_URL}/{path}".rstrip("/")
    t, desc = html.escape(title), html.escape(description)
    return "\n".join(
        [
            f'<meta name="description" content="{desc}">',
            f'<link rel="canonical" href="{url}">',
            '<meta property="og:type" content="website">',
            '<meta property="og:site_name" content="Epstein Bench">',
            f'<meta property="og:title" content="{t}">',
            f'<meta property="og:description" content="{desc}">',
            f'<meta property="og:url" content="{url}">',
            f'<meta property="og:image" content="{OG_IMAGE}">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            '<meta property="og:image:type" content="image/png">',
            f'<meta property="og:image:alt" content="{t}">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{t}">',
            f'<meta name="twitter:description" content="{desc}">',
            f'<meta name="twitter:image" content="{OG_IMAGE}">',
        ]
    )


def page(
    title: str,
    active: str,
    body: str,
    *,
    description: str = "",
    path: str = "",
    share_title: str | None = None,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{_social_meta(share_title or title, description, path)}
{FONTS}
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


def main() -> None:
    # docs/index.html is authored by hand; scripts/build_leaderboard.py bakes
    # the leaderboard tables and hero stat into it in place. This script only
    # regenerates the markdown-backed pages.
    methodology = (DOCS / "methodology.md").read_text()
    (DOCS / "methodology.html").write_text(
        page(
            "Epstein Bench: Methodology",
            "methodology",
            md_to_html(methodology),
            description=(
                "How Epstein Bench works: attribution-gated correctness, a "
                "four-stage verification gauntlet, TREC-style pooled relevance, "
                "and a pinned strong-model judge."
            ),
            path="methodology.html",
        )
    )

    card = (REPO / "dataset" / "DATASET_CARD.md").read_text()
    (DOCS / "dataset.html").write_text(
        page(
            "Epstein Bench: Dataset Card",
            "dataset",
            md_to_html(card),
            description=(
                "The Epstein Bench dataset card: task types, entity-complete "
                "corpus construction, verification, pinned models, and release "
                "statistics."
            ),
            path="dataset.html",
        )
    )
    print("wrote docs/methodology.html, docs/dataset.html")


if __name__ == "__main__":
    main()
