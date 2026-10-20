# Slides

The Module 6 defense deck: **8 slides**, the course's hard limit (a cover, 6
content slides, references).

| File | What it is |
|---|---|
| `slides.pdf` | The deck as handed in: 8 pages, every step shown |
| `slides.html` | The same deck to present from: one self-contained file, opens in any browser, with click-through steps and the speaker notes |

## Required structure, and where each part is

| # | Slide | Template requirement | Time |
|---|---|---|---|
| — | Cover | Team name, paper, topic | 0.5 min |
| 1 | The paper: problem | Baseline paper's problem | 1.5 min |
| 2 | The paper: solution | Baseline paper's solution | 2 min |
| 3 | The paper: results, and our reproduction | Baseline paper's results, with our number next to the paper's | 2 min |
| 4 | Extension 1: diagnose first (failed), and the turning point | Extension idea and result | 2 min |
| 5 | Extension 2: method | Extension idea | 1.75 min |
| 6 | Extension 2: results and conclusion | Extension result | 2.25 min |
| — | References | Full citations for the paper and the related work in the report | during questions |

Total: about 12 minutes. The numbers on the slides are the ones in
`report/REPORT.md`, and the tests recompute each of them from the committed
run files.

## Presenting from the HTML

Open `slides.html` in a browser.

| Key | Action |
|---|---|
| → or Space | next step, then next slide |
| ← | back |
| N | speaker notes (timing, what to say, likely questions) |
| F | full screen |

Fonts come from Google Fonts when online; offline, the deck falls back to
installed fonts.

## How the PDF is made

From the repository root, with Playwright's Chromium:

```
python -c "from playwright.sync_api import sync_playwright as s; p=s().start(); b=p.chromium.launch(); pg=b.new_page(); pg.goto('file://'+__import__('os').path.abspath('slides/slides.html')); pg.emulate_media(media='print'); pg.pdf(path='slides/slides.pdf', width='1280px', height='720px', print_background=True); b.close(); p.stop()"
```

The print styles put one slide on each page and show every step.

