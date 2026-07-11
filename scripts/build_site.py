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
