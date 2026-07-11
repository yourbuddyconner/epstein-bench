"""Build the corkboard for docs/board.html from build/targets.json.

Nodes are the 40 benchmark target people; an edge between two people is the
number of corpus documents mentioning both (via the release's alias index).
Layout is computed here (seeded, deterministic) so the page ships as static
markup: a yarn SVG plus pinned cards, baked between ``generated:board-viz``
markers, with an adjacency payload for the click-to-isolate interaction.

Run offline after a corpus build (requires build/targets.json, which is not
committed):
    python scripts/build_board.py
"""

from __future__ import annotations

import html
import json
import math
import re
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGETS = REPO / "build" / "targets.json"
PAGE = REPO / "docs" / "board.html"

MIN_SHARED_DOCS = 3  # drop hairline yarn nobody can see
BOARD_W, BOARD_H = 1150, 780
MARGIN = 110

# The raw alias index contains scan artifacts among the targets. This overlay
# documents the cleanup rather than trusting the index blindly:
# - two Lesley Groff fragments are one person; their doc sets are unioned
# - "morgan chase" is J.P. Morgan Chase (an organization), not a person
# - "darren ind" is a truncation of Darren Indyke
MERGE = {"who lesley groff": "lesle groff"}
EXCLUDE = {"morgan chase"}
DISPLAY_OVERRIDE = {"lesle groff": "Lesley Groff", "darren ind": "Darren Indyke"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def board_layout(names: list[str], degree: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Deterministic corkboard scatter.

    Phyllotaxis spiral ordered by connection degree (best-connected people
    pinned near the center), with a hash-seeded jitter so it reads as pinned
    by hand rather than plotted, then a few relaxation passes to keep cards
    from overlapping.
    """
    ordered = sorted(names, key=lambda n: (-degree.get(n, 0), n))
    golden = math.pi * (3 - math.sqrt(5))
    n = len(ordered)
    pos = {}
    for i, name in enumerate(ordered):
        r = math.sqrt((i + 0.6) / n)
        theta = i * golden
        # deterministic per-name jitter, no RNG state involved
        h = int.from_bytes(name.encode(), "little")
        jr = ((h % 997) / 997 - 0.5) * 0.10
        jt = ((h % 991) / 991 - 0.5) * 0.35
        x = (r + jr) * math.cos(theta + jt)
        y = (r + jr) * math.sin(theta + jt) * (BOARD_H / BOARD_W)
        pos[name] = [x, y]
    # relax: push apart any two cards closer than a card footprint
    min_dx, min_dy = 0.30, 0.16
    for _ in range(120):
        for a, b in combinations(ordered, 2):
            dx = pos[a][0] - pos[b][0]
            dy = pos[a][1] - pos[b][1]
            if abs(dx) < min_dx and abs(dy) < min_dy:
                push = 0.5 * (min_dy - abs(dy)) or 0.01
                sign = 1 if dy >= 0 else -1
                pos[a][1] += sign * push
                pos[b][1] -= sign * push
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    return {
        name: (
            MARGIN + (x - min(xs)) / span_x * (BOARD_W - 2 * MARGIN),
            MARGIN + (y - min(ys)) / span_y * (BOARD_H - 2 * MARGIN),
        )
        for name, (x, y) in pos.items()
    }


def main() -> None:
    targets = json.loads(TARGETS.read_text())
    docs: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for name, t in targets.items():
        if name in EXCLUDE:
            continue
        key = MERGE.get(name, name)
        docs.setdefault(key, set()).update(t["doc_ids"])
        labels[key] = DISPLAY_OVERRIDE.get(
            key, targets[key]["aliases"][0] if targets[key]["aliases"] else key
        ).upper()
    names = sorted(docs, key=lambda n: -len(docs[n]))

    edges: dict[tuple[str, str], int] = {}
    degree: dict[str, int] = {}
    for a, b in combinations(names, 2):
        shared = len(docs[a] & docs[b])
        if shared >= MIN_SHARED_DOCS:
            edges[(a, b)] = shared
            degree[a] = degree.get(a, 0) + shared
            degree[b] = degree.get(b, 0) + shared

    pos = board_layout(names, degree)
    max_w = max(edges.values())

    yarn = []
    for (a, b), w in sorted(edges.items(), key=lambda kv: kv[1]):
        ax, ay = pos[a]
        bx, by = pos[b]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        dist = math.hypot(bx - ax, by - ay)
        sag = min(30.0, dist * 0.12)  # yarn hangs a little
        width = 1.0 + 3.5 * (w / max_w) ** 0.5
        opacity = 0.30 + 0.55 * (w / max_w) ** 0.5
        yarn.append(
            f'    <path class="yarn" data-a="{slug(a)}" data-b="{slug(b)}" '
            f'data-w="{w}" d="M {ax:.0f} {ay:.0f} Q {mx:.0f} {my + sag:.0f} '
            f'{bx:.0f} {by:.0f}" stroke-width="{width:.1f}" '
            f'style="--yo:{opacity:.2f}"><title>'
            f"{html.escape(labels[a])} and {html.escape(labels[b])}: {w} "
            f"shared documents</title></path>"
        )

    cards = []
    for name in names:
        x, y = pos[name]
        label = html.escape(labels[name])
        cards.append(
            f'    <button class="pin-card" data-id="{slug(name)}" '
            f'style="left:{x / BOARD_W * 100:.2f}%;top:{y / BOARD_H * 100:.2f}%">'
            f'<span class="pin"></span>{label}'
            f'<span class="count">{len(docs[name]):,} docs</span></button>'
        )

    viz = (
        f'  <div class="board" style="aspect-ratio:{BOARD_W}/{BOARD_H}">\n'
        f'    <svg viewBox="0 0 {BOARD_W} {BOARD_H}" preserveAspectRatio="none" '
        'aria-hidden="true">\n' + "\n".join(yarn) + "\n    </svg>\n"
        + "\n".join(cards)
        + "\n  </div>"
    )

    page = PAGE.read_text()
    pattern = re.compile(
        r"(<!-- generated:board-viz:start -->).*?(<!-- generated:board-viz:end -->)",
        re.DOTALL,
    )
    if not pattern.search(page):
        raise SystemExit("docs/board.html: missing generated:board-viz markers")
    page = pattern.sub(lambda m: f"{m.group(1)}\n{viz}\n  {m.group(2)}", page, count=1)
    PAGE.write_text(page)
    print(
        f"baked {len(names)} cards and {len(edges)} yarn strands into {PAGE} "
        f"(max shared docs: {max_w})"
    )


if __name__ == "__main__":
    main()
