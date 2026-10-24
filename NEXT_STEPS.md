# Next steps

Work deferred on 2026-10-24, roughly in priority order. The tt-metal PR status
lives in [`INTEGRATION.md`](INTEGRATION.md).

## Install

- **Publish to the ComfyUI Registry** so the nodes install from ComfyUI-Manager.
  `pyproject.toml` already has a `[tool.comfy]` block; it needs a `PublisherId`,
  an icon, and a registry account. About half a day.
- **Prebuilt tt-metal.** The setup script still builds tt-metal from source
  (10 minutes or more, plus sudo packages). Once the pinned changes are on
  `main`, check whether a tt-metal release or the `ttnn` wheel ships the
  `models/` code the server imports; if so, the build step can go.

## Usability

- **Faster startup and model switches.** The first MiniMax-H3 load takes
  minutes, and every switch between models restarts the server. Measure where
  the time goes first, then try keeping SDXL resident next to a video model or
  making restarts cheaper. Roughly 2 to 4 days.
- **Talk tt-media-server's API (option 4).** Match tt-inference-server's
  OpenAI-style request shapes (`/v1/images/generations`, `/v1/videos/...`) for
  plain generation, so a node can point at a shared remote server such as a
  Galaxy. Keep staged ops, LoRA stacking and progress streaming as extensions,
  and offer those upstream over time. About a week.

## Communication

- **Update and send the Slack reply to Tejas.** The draft still describes four
  repos and the old branches; the install is now three checkouts plus
  `scripts/setup_tt_metal.sh`.
- **Demo reel** from the MiniMax-H3 examples (`tt_h3_cat_sunbeam`,
  `tt_h3_dragon_castle`), once the license below is settled.

## Needs someone else

- **MiniMax-H3 license.** The community license excludes the US, EU, UK and
  South Korea. Using H3 there needs MiniMax's formal application
  (https://platform.minimax.io/h3-license), filed by legal or the H3 owners
  (cglagovich, pshah, anatarajan). Until then, do not show or ship H3 output
  from those regions.

## Cleanup on qb2-120-p11t07

- Stop the old dogfood ComfyUI on port 8188 and delete `~/tt-dogfood-20261022`.
- Retire the bundled test install `~/tt-bundled-20261023` (ComfyUI on 8190)
  when it is no longer in use.
- Delete tt-inference-server's frozen `comfyui-media-server` branch once nothing
  points at it.
