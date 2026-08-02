# Screenshot-recognition MVP

Generated: 2026-08-01

## Implemented

- Windows Tk desktop application with screenshot-file and live-screen capture.
- Resolution-scaled default geometry measured from a real 2560x1440 Dota 7.41 ranked BP screen.
- Five ally and five enemy top-card slots.
- 127 CDN base portraits plus locally extracted Dota horizontal portrait variants.
- Persona/Arcana/alternate portraits mapped back to the base hero ID.
- Per-slot similarity and runner-up margin checks.
- Automatic 0v0 / 2v2 / 4v4 phase inference.
- Editable recognition results before recommendation.
- Optional two-second polling and always-on-top mode.
- Screen-only operation: no process injection or game-memory access.

The local Dota VPK was read with the open-source ValveResourceFormat CLI 19.2. Its downloaded
archive was verified against the release SHA-256 before execution.

## Real screenshot result

Input: 2560x1440 final-stage screenshot supplied by the user.

Detected ally team:

1. Anti-Mage (persona portrait)
2. Skywrath Mage
3. Warlock
4. Necrophos
5. Earthshaker

Detected enemy team:

1. Chaos Knight
2. Juggernaut (alternate portrait)
3. Jakiro
4. Shadow Fiend
5. Witch Doctor

Result: 10/10 top-card slots matched the expected heroes. This is a pipeline validation on one
screen, not a general recognition-accuracy estimate.

## Launch

Double-click `E:\dota2_bp_helper\run_desktop.cmd`, then choose either:

- `读取截图文件` to test the supplied screenshot; or
- `识别当前屏幕` while Dota is on the pick screen.

The window was smoke-tested and is currently runnable with the bundled Python environment.

## Remaining visual risks

- No incomplete-pick screenshot has been supplied yet, so empty-slot rejection has only synthetic
  coverage.
- Exclusive fullscreen may block ordinary desktop capture; borderless-windowed mode is safer.
- Non-standard HUD scaling, ultrawide layouts and multi-monitor placement may require calibration.
- New Dota cosmetic portrait variants may require refreshing templates from the installed VPK.
