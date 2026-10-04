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
- Refine engine: ComfyUI-H3-FaceRefine's tracker and stitch-back, crops
  redrawn by Wan 2.2 low-noise 14B + 4-step lightx2v LoRA
  (`comfylab_face_wan` node). Per-frame strength through core's standard
  noise mask: full at <=30px faces, none at >=120px (those frames keep their
  original pixels). Wan files (~22.5GB) download on the first refine, not at
  session start; Wan loads once per refine and is unloaded from VRAM and RAM
  before the job ends. Not audio-aware - lip sync to be judged on the first
  results.
- **Second engine, H3 (testing, 2026-10-04; job `"engine": "h3"`, test page
  Engine picker):** the pack's own H3 redraw, back with H3PerFrameDenoise in
  the path (required - it's what keeps good faces untouched: full strength
  <=30px, ZERO at >=120px, zero where the face is lost). Why it broke before:
  our ComfyUI (0.35+) applies an H3 per-frame mask natively, and the node's
  two model patches (written for older ComfyUI) stacked on top sent a
  negative timestep then NaN into H3. Fix: the sampler gets the node's
  LATENT (the per-frame mask) but not its patched MODEL. No pack fixes this
  upstream (checked 2026-10-04: Carasibana 1.1.2, Accelerated, FaceRefine-
  Plus, T8 1.89 - T8 copies the same patches and tested on ComfyUI 0.33).
  Also: audio lock rebuilt without torchaudio (ComfyLabH3AudioLock, core
  audio VAE), stitch weights from the same curve so zero-strength frames
  keep the video's own pixels (ComfyLabStrengthWeights), and
  ComfyLabH3StepCheck stops the job at the first bad timestep/NaN.
  Download: nothing - same H3 model, turbo LoRA, text encoder and VAEs the
  engine script already puts on the volume. Load: at refine time, through
  the generation graph's own loader node ids, so the H3 a generation already
  loaded is reused (no second copy); session start unchanged. Settings: base
  denoise 0.4 (pack default), 8 steps er_sde, turbo LoRA, source prompt.
  First GPU run (job 1a45f78a, 2026-10-04, cba2f8ef, 3 people): sampling
  clean - normal timesteps, no NaN, ~7-15s per person - but the worker ran
  out of RAM (89GB) stitching + 2x upscaling with H3 and its text encoder
  (~35GB) still loaded. Fixed (option A, chosen 2026-10-04): the H3 refine
  is now two ComfyUI prompts - redraw, then POST /free (unload every model,
  clear the cache), then stitch + upscale + encode, the crops handed over
  through a file on the container's own disk - ComfyUI's temp dir, faster
  than the volume (ComfyLabSave/LoadRefineCrops; deleted after each refine). Right after
  the result is delivered the session reloads H3 + text encoder (the same
  1-step warmup as session start, ~40-55s) so the next generation doesn't
  pay for it. Same run also exposed an audio-lock bug, fixed: each person's
  crop only holds the frames they're in, so H3 was hearing the clip's first
  seconds; it now gets exactly those frames' audio.
  **If we go with the Wan engine instead:** add the same auto-reload of H3
  after a Wan refine (to do, not built).
  **H3 brought up to the Wan engine's optimisations (2026-10-04)** after a
  0.6 run put a man's face on the shot-4 mother (job 848f54fc, and again on
  the original video): person 0's crop strung shot 3 (the builder) and shot
  4 (the mother) into one clip, H3 kept one face across it, and the prompt
  it redrew against was the whole 4-shot story (mostly men). Now one node,
  ComfyLabH3FaceRedraw, like ComfyLabWanFaceRedraw: one clip per person per
  shot, redrawn at <=512px, prompt
  encoded once, generic face prompt (as Wan; override still works), frames
  not redrawn keep their pixels with zero paste weight. H3PerFrameDenoise
  still sets each clip's per-frame strength (on the clip's slice of the
  tracking). Not ported: batching same-size clips - H3's mask code
  (MiniMaxH3._denoise_mask_values) uses the first batch row's mask for
  every row, so batched clips would share one clip's strengths.
  Each H3 clip covers the person's whole tracked stretch of the shot - the
  Wan engine's "small-face stretch +-4 frames" trim was tried and removed
  (2026-10-04): it saved little and cut the big-face frames H3 keeps as
  unchanged context, the identity anchor for someone walking toward the
  camera. Ruled out (user, 2026-10-04): giving H3 a reference photo of the
  person's face.
  **Test results and decisions (2026-10-04, all on cba2f8ef, seed 424242):**
  - Strength: 0.4 for H3 (0.5/0.6 look bad). The pack ships 0.4; on H3's
    shift-12 schedule 0.4 already redraws ~89%, 0.6 ~95%.
  - Clips: one per person (whole track) is the default. Per-shot clips made
    faces clearly worse (3b749775 per shot vs 36c1d9a7 per person, 0.4);
    why is still open. Kept as a test option (job split_shots: true).
  - Prompt: the generic face prompt stays. The man's face on the mother came
    from the whole-scene prompt, not the cut: 0.6 with one clip per person
    and the generic prompt showed no man's face.
  - Known side effect, accepted for now (user: "talking isn't a big deal"):
    with the generic prompt the shot-3 builder's mouth moves when nobody is
    talking (too fast, jaggy). Not the audio - 32fb5f03 used the same
    frame-aligned audio with the scene prompt and his mouth stayed still;
    the scene prompt said the narrator is off-screen, the generic one says
    nothing. If it needs fixing: send H3 only that shot's part of the scene
    prompt (needs detected cuts matched to the prompt's shot numbers).
  - The girl's face (smallest, 23-29px) flickers back to melted near the end:
    the detector misses it on a few frames (308, 353-356, 360). Detector at
    960px instead of 640 (face_detector.py; a 23px face reached it at ~11px,
    then ~16px): her missed frames 6 -> 1, no measurable time cost (8.5-13s
    vs 8.6-12s), but it also found a 16px "face" on ~5 frames near 311 that
    the counter makes a 4th person. Bumped to 1280 (user, 2026-10-04) - watch
    for extra faces. Both engines.
  - Wan steps: 4 is the default again (user, 2026-10-04). Redraw time on
    cba2f8ef at 0.6: 4 steps 90.5s, 3 steps 82.6-93.1s, 2 steps 66.0s (2
    looked okay; ~20s saved, ~10% of the refine). 3/2 stay as test options.
  - H3 steps: 8 (8-step turbo LoRA, default), or 4/3/2 with the 4-step
    768p turbo LoRA - to be compared (roadmap step 1).
  - Decision: if H3 is chosen, the Wan engine gets removed.
  - **Face-size range retuned (user, 2026-10-04):** ignore faces under 22px
    (not counted, tracked or pasted - the faces test video's aerial shot
    had 5-10px "faces" redrawn at full strength); full strength up to 45px,
    none from 60px (was 30 -> 120). Calibrated on: Alpha Timber mother and
    daughter 23-31px (fix, smallest worth fixing); faces test video
    (813cdb5d) friends ~57-64px (fine, don't touch) vs the middle one
    ~52-58px (fix). Those last two are only a few px apart, so this edge is
    fragile; the face count now logs each shot's repairable face sizes
    ("repairable faces ~59px, 55px, ...") to tune it.
  **Decision (user, 2026-10-04): no automatic face fixing - "Fix faces"
  stays a button the user presses after generation.** Face size is not a
  reliable sign that a face is broken: on the Shrek video (51245f65, job
  3b73aab9, Wan 0.6 / 4 steps) Lord Farquaad's ~27-111px face already
  looked fine and the refine degraded it. Roadmap step 3 (auto-fix during
  generation) is dropped; the faces-at-every-distance test is still useful
  for the default size range.
  Same job: both tracked "people" had identical tracks (175 frames, face
  27-111px, mean strength 0.65) - two trackers on Farquaad's face, so he was
  redrawn and pasted twice. The duplicate-tracker check (drop a person whose
  face centre sits on an earlier person's for >half the shot) was declined
  when this only showed on a refine-of-a-refine; this was an original video.
  Built (user go-ahead, 2026-10-04): comfylab_face_wan/duplicates.py, used
  by both redraw nodes. On source frames where both trackers detected a
  face, same face = both land on the same detector face box (the detection
  inside the person's crop box nearest their face size and the box centre;
  face_pick from H3 Load Video + Face Select) - not the crop boxes, whose
  centre drifts off the face when clamped at the frame edge. Fallback
  without detections: centres closer than half a face height. Same face on
  >50% of those frames -> the later person is dropped (not redrawn, zero
  paste weight); less -> kept, nothing pasted on the overlapping frames.
  Reports say "person N follows person M's face ... duplicate".
  Stylised/cartoon videos: the generic face prompt asks for a realistic face.
  Before production, gate the button on the video's style - most reliable:
  the prompt enhancer outputs a fixed style label (photoreal / animated)
  stored with each generation; fallback for videos without one: a quick
  image check (CLIP) on a few frames, hiding the button when unsure.
  **"Repair faces" button - site UX requirements (user, 2026-10-04; not
  built):**
  - Obvious right after a generation is served, so a user who sees broken
    faces immediately knows a fix exists rather than being put off.
  - Also a clear option in the site's built-in CapCut-style editor.
  - In the editor it must be obvious from how it's integrated (not from
    explanatory text) that it applies per clip, not to the whole edit -
    e.g. attached to the selected clip on the timeline.
  - A question-mark icon next to it with: "Minimax H3 may struggle on
    smaller faces, notice glitched faces? Repair using this tool 2-5 mins".
  - Pressing it shows a dropdown of styles to pick before it runs; the
    chosen style is inserted into the generic face prompt. Realistic is the
    default. (Ties in with the style-gating note above: a style label saved
    at generation could preselect the dropdown.)
  **To do (noted 2026-10-04, not built): Stop GPU must kill instantly -
  force kill, whatever is running.** Today a job in progress is force-killed
  on Stop (should_force_kill), but the worker's own warmup generations are
  not: the post-H3-refine reload (~47s), the session-start warmup and the
  re-warm after a ComfyUI crash all call run_generation without
  should_force_kill, so a Stop during one waits for it to finish - and that
  GPU time is billed. Pass the same session-stopped check to all of them.
  **To consider (noted 2026-10-04, not built): reload H3 after a refine
  without a warmup generation.** Today the reload runs the session-start
  warmup (a 1-step 320x320 generation, ~47s in job 10e921bd's session) just
  to get H3, the text encoder and the VAEs back into memory. A load-only
  prompt (the loader nodes plus a minimal output node, no sampling) would
  skip the throwaway generation. Check first: in the logs H3's "Model
  Initialization" (~22s) happens at the first sampling step, not at load,
  so a load-only reload may leave that part for the next generation - time
  both and compare before switching.
  Later (asked 2026-10-04, not built): a 4-step option using Comfy-Org's
  4-step turbo LoRA (minimax_h3_fl2v_turbo_4step_v1.0_768p, LORA_CHOICES
  "fast" - already on the volume, nothing to download). Roughly halves the
  H3 redraw time; trained for 768p, and crops are capped at 768. Build only
  after the 8-step H3 refine is confirmed working, then compare same seed.
- **"Fix faces" button settings (chosen 2026-10-03, test job 96cb3da1 on
  cba2f8ef; steps changed 3 -> 4 on 2026-10-04):** strength (denoise) 0.6, 4 steps, Standard crop size (tracker
  canvas, redrawn at up to 512px), people counted automatically (up to 4
  per shot). All are the worker's defaults, so the site only needs to send
  `mode` and `source_video_key`.
  Test finding (2026-10-03, job d6128f80 vs earlier 4-step runs on the same
  clip): 3 steps looked as good as or better than 4, so 3 is a quality
  choice, not just a speed one. Untested guess at why: fewer steps on an
  already-detailed crop means less over-smoothing ("waxy" look).
- **Refine speed ideas, not built yet (decide once the per-step timings are
  in - the job output now has `timings`):**
  - Cache the native-size (pre-upscale) copy of every generation on the
    volume when it's made, so a refine skips downloading, decoding and
    shrinking the upscaled video (~10-20s, needs a small generation change).
  - Done for the refine: GPU H.264 (ComfyLabSaveVideoNVENC, h264_nvenc
    cq 19, falls back to core's libx264 crf 18). Needs the image's
    NVIDIA_DRIVER_CAPABILITIES to include "video". Generation still uses
    core SaveVideo (CPU) - switch it too once the refine confirms NVENC works.
  - One stitch for everyone (one pass over the frames instead of one per
    person, touching only the face areas).
  - Don't re-upscale the whole video: paste the refined faces into the
    original upscaled video frame by frame outside ComfyUI (doing it inside
    ComfyUI needs ~18GB RAM per copy at 2x for 15s).
- **Wan refine to-dos (noted, not built - after the current test):**
  - Done: auto people count. A short first ComfyUI prompt runs face
    finding + ComfyLabSmallFaceCount (the most faces under 120px on screen
    together in any shot, capped at 4 - the largest small faces win), then
    the refine is built with exactly that many; the face finding is reused
    from ComfyUI's cache. 0 = "no small faces", original video returned.
    Person i = the i-th largest SMALL face per shot (already-big faces are
    skipped). Job input subjects is optional (1-4 still accepted).
  - Drop the prompt override for Wan: always use the node's generic face
    prompt (one override applies to every person, and the source prompt is
    a whole-scene H3-format prompt that never goes to Wan).

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
- **Consider later: freeze the Wan refine downloads to a fixed version.**
  `ensure_model_file` in `handler.py` downloads the Wan files from Comfy-Org's
  repos' `main`. If a file is re-uploaded, a fresh volume silently gets the
  new version. Pinning each URL to its HuggingFace commit (`resolve/<sha>/...`)
  would make results reproducible. Not done yet.
- **Done: removed the fine-tune bake-off and HIGHBITRATE VSR.** DaSiWa V3,
  Singularity v1.3 and the fal Realism People LoRA (none changed the melted
  faces) are gone from the code, preload and test page; delete their files
  from the volume by hand. The HIGHBITRATE_ULTRA RTX VSR option and its custom
  node are gone too; the standard RTX VSR (ULTRA) upscale stays.
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

## Wan strength by face size (2026-10-04)

The user's call: 0.3 looked best on most faces, but it was too gentle on the
Alpha Timber mother (23-30px) and daughter (20-24px), where 0.6 was much
better. So each Wan clip (one person in one shot) now gets one strength,
picked from that person's typical (median) face height in that shot:

- under 32px (`REFINE_FACE_PX_TINY`): 0.6 (`WAN_REFINE_SMALL_FACE_DENOISE`, job input `small_denoise`)
- 32px and up: 0.3 (`WAN_REFINE_DEFAULT_DENOISE`, job input `denoise`), still fading to nothing at 60px

The boundary sits under the father (~27-45px, mostly upper 30s) and above the
mother and daughter. It's a real 0.3 or 0.6 pass per clip, not a blend. Clips
with different strengths are never batched together. Each clip's report line
shows its strength.

The floor dropped from 22px to 18px (`REFINE_FACE_PX_MIN`, shared by both
engines) because the daughter measured 20-24px, so 22px missed her on part of
the shot. H3's strength is unchanged (0.4, its own per-frame curve).

## Park video: 0.3 and 0.4 looked the same (2026-10-04)

The refine report on 813cdb5d explained it. The friends in shot 2 measured
56-59px, inside the old 45->60px fade, so they got a partial pass or none and
the strength barely mattered. One tracker landed on a 15px background face
(then thrown out by the 18px floor). The best-tracked friend was only detected
on 52 frames, and the refine fades out wherever there's no detection. The user
asked for all three fixes; each is a job input / test-page option with the old
value one click away:

1. Size range: full strength up to 60px, untouched from 62px (was 45/60).
   First set to 60/90, but the Wan paste is full once the fade's strength
   reaches 0.2 (`WEIGHT_RAMP`), so 60/90 really redrew everything up to ~84px
   and would have changed good faces. 60/62 keeps the old 60px line: the three
   friends at 56-59px get the full pass, and the furthest-right one (60px+),
   who looked fine, stays untouched. Watch for faces hovering at 60-62px
   switching between redrawn and not.
   `REFINE_FACE_PX_SMALL`/`_LARGE`, job inputs `face_px_small`/`face_px_large`.
2. Crop: 2x the face instead of 3x (`REFINE_CROP_FACTOR`, job input
   `crop_factor`). The face fills half the crop, about 1.5x the pixels to
   redraw, with less context around it. The stitch's soft edge scales with it
   (`stitch_feather`: 24px at 3x, 16px at 2x). Unscaled, a 28px face at 2x
   left ~3% of the redraw showing at the crop border.
3. Detector bar 0.35 -> 0.25 (`REFINE_DETECT_CONFIDENCE`, job input
   `confidence`): keeps turned, blurred and shadowed faces the old bar dropped.
   More junk clears it too, which the 18px floor and duplicate check catch.

All three apply to both engines. The 32px small-face strength split (Wan) is
unchanged.

## Bug: the park friend in pink was skipped (fixed 2026-10-04)

On 813cdb5d (job 7dd06a26) the count found four faces to fix in shot 2
(61/60/58/56px), but persons 2 and 3 both locked onto a 16px background face,
so one friend was never redrawn. `ComfyLabFacePickIndex` ranked among ALL faces
on its lock frame, skipping the shot's "large face" count first. That count is
the most large faces on screen at once anywhere in the shot, not on the lock
frame. With faces hovering at the 62px line, the skip overshot past the
friends onto tiny background people. Now (`_small_face_lock`) it ranks only
the in-range faces (18 to under 62px), and everyone in a shot locks on the
same frame, the one with the most in-range faces, so no two people start on
the same face. Both engines use this node.

Then 60/62 -> 60/65 (jobs 3e107710 / 75075b9e, 0.3 and 0.4): with the pick
fix all four friends were redrawn, but the woman on the left (59-63px) was 62px+
for the first ~16 frames of her shot, so those kept the original face. 65 covers
her max of 63 (the paste is full up to ~64px).

Possible default change (user, 2026-10-04, not decided): Wan strength 0.4
instead of 0.3 for faces 32px and up. 0.3 and 0.4 both looked good on the park
video once the faces were inside the range. Small faces stay at 0.6.

## Bug: the Alpha Timber daughter was skipped after the pick fix (fixed 2026-10-04)

The pick fix above also locked everyone in a shot on the frame with the most
in-range faces. On Alpha Timber (jobs c56dbedf / a3f12049 / efca951f) that
was frame 329, near the end of the family shot, where a stray 4th small face
appears. Tracked back from there, the father's tracker lost him after frame 330
and the daughter's slot sat on a face detected on only 13 frames, so she was
effectively never fixed. Before the fix (d48baabc) the lock was frame 264, the
shot's start, and she was tracked on 94 frames. Now each person locks on the
first frame of the shot holding that many in-range faces (the pack's own
rule), still ranked over in-range faces only, so the park fix holds.
If two people land on one face, the duplicate check drops the later one.

## Ghosting on the family shot: bridging short detection misses (2026-10-04)

The Alpha Timber family shot (job 5103637e) was good on some frames and
ghosted on others. The paste weight is the tracker's "face detected this
frame", smoothed, so each frame the detector misses on a small face dips the
paste to ~40-80% for a few frames. That blends the redrawn face with the
original melted one, which shows as a pulsing ghost. The daughter missed
frames 268, 311-312 and 314. `gaps.py` now treats gaps of up to 8 frames
(1/3 s) inside a shot, with a detection on both sides, as detected. Weights
are only raised, so real dropouts (face turned away, left the shot) and shot
edges fade exactly as before. Both engines; the subject line in the report
says how many frames were bridged.

## Family clip: father skipped, faces lost at the end (fixed 2026-10-04)

Job 3d29b7fd on da2f6e17 (the 7s family clip):

1. Person 2 landed on person 0's face, so the duplicate check dropped it and
   the father was never fixed. The three faces are ~29/29/26px, so the size
   order shuffles frame to frame, and "3rd largest on frame 3" was person 0's
   face. People are now settled in order: person j still locks on the first
   frame holding j+1 in-range faces, but takes the largest face there that
   persons 0..j-1 haven't got (their faces followed forward from their own
   lock frames). Side effect on Alpha Timber: the 4th slot now goes to the
   stray small face that appears late in the family shot (possibly the
   reflection in the glass door) instead of doubling up on the daughter.
2. All three faces shrink to 15-17px near the end, under the 18px floor, so
   those frames kept the melted original. 18px still decides who counts and
   gets picked as a person (background people stay out), but a person being
   tracked is now fixed down to 12px (`REFINE_FACE_PX_MIN_TRACKED`). Risk: a
   tracker that drifts onto a tiny background face mid-shot would get it
   redrawn.

The 8-frame gap fill (`gaps.py`) stays, by the user's choice.

## Official face-refine settings for the port (user, 2026-10-04)

Signed off after the park (813cdb5d), Alpha Timber (cba2f8ef) and family clip
(da2f6e17) runs:

- Engine: Wan 2.2 low-noise + lightx2v, 4 steps
- Strength: 0.3 per clip; 0.6 for clips whose face is typically under 32px
- Face range: full up to 60px, untouched from 80px (paste full up to ~76px)
- Crop: 2x the face; detector bar 0.25 at 1280px
- Floor: 18px to count/pick a person; a tracked person is fixed down to 12px
- Picking: in-range faces only, in order, no two people on one face
- Detection gaps of up to 8 frames inside a shot are bridged
- Duplicate check on; generic face prompt; button-only, never automatic

## Start/end frames and reference images (2026-10-04)

Start/end frames: wired on the test page (`startFrame`/`endFrame`), using
the worker's existing `start_frame`/`end_frame` job fields on
`MiniMaxH3ImageToVideo` (the current fl2va model supports both). The page
centre-crops each image to the generation's aspect and sizes it to exact
width x height before sending: H3 stretches the first frame to the canvas, so
an uncropped image would come out distorted. A job with a reference image AND
start/end frames is now refused (user: one or the other per generation).
Untested on a real generation yet.

Reference images (user: up to 9, `<Picture N>` in the prompt): NOT built yet.
They need H3's separate ref2va checkpoint (`minimax_h3_ref2va_pruned_int8_convrot`)
and its own turbo LoRA (`minimax_h3_ref2v_turbo_4step_v0.1`). The installed
fl2va model only does text and start/end frames. Pending the user's call on
download, load and RAM. The worker's existing single `ref_image` path would
run reference conditioning on the fl2va model, which isn't what it's for.

## DaSiWa Hybrid V3 back as a per-session model (test, 2026-10-04)

Test page "H3 model for this session" dropdown, sent as `model` with the
session-start job. `run_session` downloads it to the volume if missing (21GB,
gated repo, uses the endpoint's HF_TOKEN), then the warmup, every re-warm,
the post-refine reload, every generation without its own `model`, and the H3
refine engine all use it, so a session only ever has one H3 model loaded.
Purpose: an aesthetics A/B against base (the only earlier comparison,
cba2f8ef vs 7f224640, differed in attention mode too), turbo compatibility
(never tested on DaSiWa), then start/end frames and reference images. If it
wins, it can be the single model and reference images need no second
checkpoint or swapping.

## Decision: base stays the text-to-video model (user, 2026-10-04)

After the blind A/B (Batman too close to call; fisherman close-up DaSiWa by a
landslide; cycling base narrowly, "cleaner, less movement" vs DaSiWa's "more
motion, actually pedalling") and the Flash/Dora rerun of the user's favourite
base generation (a9f8afdc) on DaSiWa, the user called it: base wins for
text-to-video, "no contest". Base stays the default for text and start/end
frames. Reference images go ahead on the swap plan: a reference job loads a
reference-capable model and the next normal job swaps back. Which model that is
(official Ref2VA or DaSiWa) is still open; a RunPod pod test of Ref2VA with the
Fun ControlNet Union template (motion transfer, Higgsfield Genjutsu-style) is
under way and will inform it. The DaSiWa session option stays on the test page
for now. Not built yet: the idle-timer fix (idle clock should start after the
warmup, not before a first-time download) and the reference-image UI.

## To test later: Ref2VA for plain text-to-video (2026-10-04)

Nobody has published a head-to-head of Ref2VA with no references against FL2VA
for text-only generation. All guidance says "use FL2VA for text", but no tests
back it either way. Comfy-Org's own Fun ControlNet template runs Ref2VA with
empty reference slots and calls that text-to-video, so it works. If Ref2VA's
text-only output matches base, one model could cover text AND references with
no swapping (start/end frames would be the one gap: Ref2VA's node has no
first/last-frame slots). Test on the RunPod pod: Flash/Dora prompt, seed
424242, 1280x720, 15s, 20 steps, no references, ControlNet bypassed, compared
against the user's favourite base run a9f8afdc.

## To try tomorrow on the RunPod pod: Viggle-Animate and SCAIL-2 (2026-10-04)

Character swap / motion transfer (Higgsfield Genjutsu-style), for music-video
recreations. Same clips through both, timed and judged blind:
1) one dancer, 5s; 2) a character bigger than the person (panda/hippo over a
human); 3) two people (e.g. a rap video, rappers -> hippo + lion).

- Viggle-Animate: Viggle's open H3 Ref2VA finetune. Inputs: driving video +
  one frame of it repainted with the new character(s). No mask, skeleton or
  prompt. ComfyUI: Saganaki22/ComfyUI-Viggle-Animate-H3 (install from the
  original repo, not the goofyrodent mirror), weights from
  drbaph/Viggle-Animate-ComfyUI: pruned int8 (21GB), dmd LoRA r64 (0.94GB),
  precomputed text embed (no 15GB text encoder), video VAE. 4-8 steps, cfg 1.
  Measured elsewhere: ~33s per 5s on a 5090. Multi-person "feasible, not
  robust"; weak lip-sync in close-ups.
- SCAIL-2 (zai-org, Wan 2.1 14B): official template
  video_wan21_scail2_character_replacement. All nodes are already in pinned
  ComfyUI fb2315f, no custom packs. Inputs: driving video + any character
  image; SAM3 finds people by text ("human"), multi-person assigned left to
  right. Use the int8 model (16.7GB), not the template's fp16 (32.8GB). 6 steps
  + lightx2v 0.8 + DPO LoRA, 896x512, 81-frame segments (76 new + 5 overlap),
  each queued by hand; frames are taken 1:1 from the source (30fps source ->
  2.7s per segment). Estimate (unmeasured): ~70-90s per segment on a 5090.
- Record "Prompt executed in" for every run; replace the estimates above.
