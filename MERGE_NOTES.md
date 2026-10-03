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
  Rules and test log: PROMPT_FRAMING_RULES.md.
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
- Two refine engines on the test page (job input `engine`):
  - `h3` - ComfyUI-H3-FaceRefine as shipped. Its H3PerFrameDenoise (what
    keeps large faces from being redrawn) breaks sampling on our ComfyUI
    (issue #19 on the pack; works on 0.34, broken on 0.36/0.38).
  - `wan` - same tracker and stitch-back, crops redrawn by Wan 2.2 low-noise
    14B + 4-step lightx2v LoRA (`comfylab_face_wan` node). Per-frame strength
    through core's standard noise mask: full at <=30px faces, none at
    >=120px (those frames keep their original pixels). Wan files (~22.5GB)
    download on the first Wan refine, not at session start; Wan loads once
    per refine and is unloaded from VRAM and RAM before the job ends. Not
    audio-aware - lip sync to be judged on the first results.
- **At merge, if Wan is the engine we keep: remove the H3-only refine
  pieces.** Startup cost today is small (no model loads, no downloads at
  session start), but these exist only for the H3 engine:
  - `MiniMaxH3NativeAudioLock` (Dockerfile copy from the Shrek3OnVH5 repo) -
    the one H3-refine piece imported at every ComfyUI start, and it pulls in
    torchaudio;
  - the torchaudio pin and the libgl1/libglib2.0-0 note tied to it (keep
    torchvision's pin; check nothing else imports torchaudio first - core
    ComfyUI dropped it);
  - `comfylab_debug` nodes (copied every boot) and the debug wiring in
    `build_refine_payload`;
  - `build_refine_payload`, `REFINE_DEFAULT_DENOISE`/`REFINE_STEPS`, the H3
    option on the test page.
  Keep for Wan: the ComfyUI-H3-FaceRefine pack (its tracker and stitch are
  used; its nodes import lazily, ultralytics only when a refine runs),
  ultralytics/scipy/scenedetect, face_yolov8m.pt.

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
- **Done: restored an unsaved koboldcpp video.** Job `4ba19710` on the live
  site (2026-10-03, Alpha Timber prompt) produced
  `ad4ddca3-f097-41aa-ba2b-da1ffbb69d5c.mp4`, but the save was blocked
  because the owner's account has no subscription (library cap 0). At the
  owner's request, a `generations` row (`a449da96...`) was inserted by hand
  so it shows in the Library. The row didn't go through the cap check, so
  that account now holds 5 saved videos against a cap of 0.
- **Unsubscribed accounts silently lose generations.** While billing isn't
  launched, every account without a subscription row has a library cap of 0,
  so every result is generated and then can't be saved, including test
  accounts. Fix as part of the library-cap decision below: give test
  accounts a cap or a test subscription row, and/or warn before generating
  that the result can't be saved.
- **Undecided: library caps.** `cleanup.js` (Railway "cleanup crew", hourly)
  rolls every account's library over at a flat 30 videos and deletes anything
  older than 15 days. That's intentional per the owner. But `server.js`'s
  per-tier caps (Starter 15 / Creator 40 / Business 100) mean Creator and
  Business can never actually exceed 30. Decide whether `cleanup.js` should
  respect each tier's cap. Either way, `minimax-h3-website/CLAUDE.md`'s
  "Library caps" section is out of date and needs correcting.
