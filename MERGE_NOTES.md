# Merge notes: ComfyLabV2 -> production

Decisions and to-dos to carry over when this worker replaces production's
koboldcpp worker (`minimax-h3-worker`) behind `minimax-h3-website`.

## Decided

- **Upscaler: NVIDIA RTX VSR 2x only.** Folded into the same ComfyUI submission
  as the generation (no `/free`), so MiniMax H3 stays resident between jobs.
  Confirmed clean on a real run 2026-10-01. ESRGAN 2x and FlashVSR were tested
  and removed.
- **LoRA: only "Turbo" (8-step)**,
  `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, verified on the
  test page. At merge, remove every other LoRA from production:
  - production worker `LORA_CHOICES` (`quality` ->
    `minimax_h3_turbo_ema_ckpt500`, `fast` -> `minimax_h3_lightx2v_turbo`)
    and their cold-start downloads
  - the frontend presets (`web/src/components/generate/constants.ts`,
    `GeneratorPanel.tsx`, `settings/GenerationTab.tsx`)
  - this repo's `fast` (4-step 768p) preset and its download in
    `ensure_comfyui_engine.sh`

## Must do at merge

- **Heartbeat -> production reaper.** `touch_session_heartbeat()` and
  `mark_session_worker_started()` write to `comfylab_active_gpu_sessions`
  (filtered by `slot = 'default'`). Repoint them at `active_gpu_sessions`
  (filtered by `session_id`) so `server.js`'s `reapDeadWorkers` and the billing
  meter see them. Both writes fail silently, so on the first merged test,
  verify that `last_activity_at` actually advances.

## Known limitation: faces at medium/long range (pinned)

Base MiniMax H3 renders faces poorly once a head is a small part of the frame
(medium and wide shots). Confirmed by our own tests, 2026-10-02: base, base +
fal Realism LoRA, DaSiWa Hybrid V3, 7s vs 15s, Sage on vs attention off - all
bad. So it's not Sage, not clip length, not the fine-tune. The community
agrees it's a property of head-size-in-frame, not output resolution
(Comfy-Org/MiniMax-H3 HF discussion #30, ComfyUI-H3-FaceRefine README).

- Ship a user-facing disclaimer: face detail drops in medium and wide shots;
  close-ups look best.
- Bias the prompt enhancer toward close and medium-close framing for people.
- A real fix is a second pass (face-crop refine, v2v detailer, or MiniMax's
  hosted Regenerate-2K); still being evaluated.
- Feature idea (agreed direction, build only once the refine is proven on
  the test page): an opt-in "See glitched faces? Refine faces" action on a
  finished video. Run a face detector on the result (seconds) and only offer
  it when there are small faces (under ~120px tall), with a time estimate
  per person ("We spotted 3 small faces - refine? ~5 min"). Runs as a normal
  job on the user's GPU session, so it's metered like any generation; if the
  GPU is stopped, the button says it'll start a session first. Needs: keep
  the raw 768p output + prompt + seed per generation (refine works pre-
  upscale, then re-upscales), keep both original and refined versions
  (decide how that counts toward library caps). Test-page version:
  handler.py run_face_refine / "Refine faces" panel.

## Other to-dos (not merge-blocking)

- **Tune the prompt enhancer to MiniMax's prompt guide.** The live site's Claude
  prompt enhancer (`POST /api/enhance-prompt` in `minimax-h3-website/server.js`,
  UI in `web/src/components/generate/PromptAssistant.tsx`) should follow
  MiniMax's own guide:
  `https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`.
  MiniMax says its prompt pre-processing (H3-Context-IR) "is critical to the
  quality of the final output". Key rules from the guide:
  - Every detail must be something visible or audible; no abstract
    instructions or mood words.
  - Open `[Shot 1]` with the style and composition.
  - Number later shots, each with a cut time (`[Shot 2] At 00:03.500, the
    camera cuts to...`); prefer camera motion over a cut for small changes.
  - Give each speaker or character a stable identity on first appearance.
  - Don't overload short clips: too many characters/events in ~7s makes
    faces small and glitchy.
- **Consider later: freeze bake-off model downloads to a fixed version.**
  `ensure_model_file` in `handler.py` downloads DaSiWa V3, Singularity and
  the fal LoRA from each repo's `main`. If an author re-uploads a file, a
  fresh volume (or a re-download) silently gets the new version. Pinning
  each URL to its HuggingFace commit (`resolve/<sha>/...` instead of
  `resolve/main/...`) would make results reproducible. Matters most once a
  model is chosen for production. Not done yet.
- **Undecided: library caps.** `cleanup.js` (Railway "cleanup crew", hourly)
  rolls every account's library over at a flat 30 videos and deletes anything
  older than 15 days. That's intentional per the owner. But `server.js`'s
  per-tier caps (Starter 15 / Creator 40 / Business 100) mean Creator and
  Business can never actually exceed 30. Decide whether `cleanup.js` should
  respect each tier's cap. Either way, `minimax-h3-website/CLAUDE.md`'s
  "Library caps" section is out of date and needs correcting.
