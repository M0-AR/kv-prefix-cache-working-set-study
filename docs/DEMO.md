# Demo video / GIF guide

The highest-conversion asset for a repo like this is a 12–60 second loop
showing one complete value flow in a real terminal. Our flow:

```bash
./scripts/demo.sh
# -> 6 experiments, live BurstGPT fetch, ALL EXPERIMENTS PASSED
```

## Option A — scriptable and reproducible (recommended): VHS

[VHS](https://github.com/charmbracelet/vhs) records from a `.tape` file, so the
demo regenerates with one command. Install, then:

```bash
# docs/demo.tape (already in this repo) -> assets/demo.gif + assets/demo.mp4
vhs docs/demo.tape
```

Terminal geometry 80x24 (readable on mobile), dark theme, 14–16pt font,
short `$ ` prompt, warm the BurstGPT cache with one unrecorded run first so the
GIF shows the demo, not a download spinner. Keep it under 2 MB for the README
embed (15 fps, 64-color palette, max 1200px wide).

## Option B — asciinema + agg (copy-pasteable playback)

```bash
asciinema rec assets/demo.cast
./scripts/demo.sh
# stop recording, then:
agg assets/demo.cast assets/demo.gif
```

Embed the GIF in the README; embed the `.cast` with
[asciinema-player](https://github.com/asciinema/asciinema-player) on the
interactive site so visitors can copy commands from the player.

## Option C — quick GUI pass: ScreenToGif / Loom

Point-and-click recording with trim, text overlays (`<- 122 pages @70%`),
and crop. Add the callout on the final held frame and keep the last frame
visible ~7 seconds so readers absorb the payoff.

## Embed checklist (verify before push)

- Commands shown match the current CLI (re-run right before recording).
- No personal paths, secrets, or stack traces in frame.
- Loops cleanly; readable at README width and on a phone.
- GIF < 2 MB (GitHub hard cap 10 MB); MP4 version for social/docs.
- `assets/demo.gif` committed; raw takes gitignored.

## Current status

`preview.html` ships an **animated in-page demo** (no download needed):
watch K/V reuse, prefix hits, and LRU eviction play out step by step.
A recorded GIF/MP4 is a welcome contribution — see `docs/demo.tape`.
