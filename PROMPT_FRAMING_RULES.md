# Prompt framing rules for MiniMax H3 (people and faces)

Working rules for writing H3 prompts, and the source for the site's Claude
prompt enhancer (`/api/enhance-prompt`, `PromptAssistant.tsx`) at merge. Goal:
agency-grade people shots without needing a refine pass.

Every rule is tagged with how much we actually know:

- **[tested]** we saw it in our own runs (see the test log at the bottom)
- **[community]** reported by other H3 users, not yet checked by us
- **[hypothesis]** our reasoning, still to be tested

## Why framing matters

Base H3 draws faces badly once a head is a small part of the frame. This
depends on how tall the head is in pixels, not on output resolution, so
upscaling doesn't fix it. **[tested]** It's not Sage attention, not clip
length, and not the fine-tune: base, base with the Realism LoRA, DaSiWa V3,
7s and 15s clips, and Sage on/off all produced the same melted faces. It's
not our ComfyUI stack either: production koboldcpp (different weights,
engine and attention code) melted the same framings.

H3's native output is 768px tall. ComfyUI-H3-FaceRefine's defaults put
"broken" at 30px of face height or less and "fine" at 120px or more.
**[community]** In our failed family shot, heads were about 45–55px tall at
768p. **[tested]**

**Working target: every visible face should be at least ~120px tall at 768p,
about 1/6 of the frame height or more.** **[hypothesis]** The threshold is
borrowed from the refine pack. Tighten it once we've measured our own
results.

## Rules

### 1. Frame every person whose face is visible chest-up or waist-up

- Use explicit body cut-offs, not shot-size jargon alone: "a medium
  close-up, framed from the chest up", "a medium shot, framed from the
  waist up".
- Add "face clearly visible, large in frame".
- **[tested]** Shot 2 of the original Alpha Timber ad (a close-up of one
  tradie, camera at chest height) came out well. Close and medium-close
  framing of one person holds up.
- **[hypothesis]** Explicit body cut-offs constrain H3 more than "medium
  shot" does. Not yet tested.

### 2. Never pair visible faces with wide framing words

- In any shot where a face is visible, avoid: "wide shot", "medium-wide",
  "full shot", "long shot", "establishing shot", "full body", "head to toe".
- **[tested]** Shot 3 ("a medium-wide shot … a builder … gives a
  thumbs-up") melted the builder's face. Shot 4 (a family stepping out,
  ending on "a static wide shot of the home and deck") melted all three
  faces.

### 3. Wide and establishing shots: no visible faces

Wide shots are fine as long as no face has to be rendered. Use one of:

- no people at all (product, location, building);
- people seen from behind, in silhouette, or turned away ("seen only from
  behind");
- hands and details only (hands on timber, tools, product close-ups).

**[tested]** Shot 1, the warehouse wide shot with a forklift and no
foregrounded faces, was fine.

### 4. Camera moves must not end wide on people

- **[tested]** In Shot 4, "the camera pulls out with large amplitude … and
  settles into a static wide shot" turned a medium shot of the family into
  a wide, and the faces melted.
- **[hypothesis]** On people:
  - prefer push-ins, which make faces bigger;
  - prefer trucking moves at a fixed distance and static shots;
  - keep any pull-back small, and state where it ends ("settles into a
    medium shot framed from the waist up").

### 5. Keep group shots small and tight

- **[hypothesis]** More people side by side means a wider frame, which
  means smaller heads.
- Cap groups at 2–3 people. Bring them close together with heads at a
  similar height, for example a child held in an arm or people leaning in,
  and frame waist-up.
- Larger groups (crowds, teams, audiences) should be shown from behind, out
  of focus, or in separate shots. See rule 5a.

### 5a. Crowds and large groups

**[hypothesis]** None of this is tested yet. Every crowd face can't be
refined: the refine pass is capped at 4 people and is meant for the main
faces. So a crowd shot has to be written so that background faces are never
read as faces.

- **Foreground/background split.** Put 1–3 people in focus in the
  foreground, framed chest-up, with the crowd behind them softly out of
  focus: "the crowd behind them is out of focus, faces indistinct".
- **Turned away.** Show the crowd seen from behind: marching away from
  camera, facing a stage, looking toward a building.
- **Hands and objects instead of faces.** Close-ups of signs, raised fists,
  flags, banners, feet marching, hands clapping.
- **Separate shots.** A wide establishing shot with no readable faces
  (from behind, from high above, or in silhouette), then cuts to tight
  single-person shots.
- **Light that hides faces.** Silhouettes against a bright sky, backlight
  at dusk, or smoke and haze, so faces are naturally unreadable.
- **Avoid:** "a crowd of people looking at the camera", "faces in the
  crowd", or a sharp, front-facing wide crowd shot. That's guaranteed to
  melt every face.
- If a crowd shot still has a few sharp mid-ground faces, refine only the
  2–4 the viewer will look at.

### 6. Put faces in focus and in good light

- **[hypothesis]** Shallow depth of field on the subject ("background
  softly out of focus") puts the detail where the face is.
- **[hypothesis]** Soft front or side light ("well lit by soft afternoon
  sun") helps H3 resolve features. Faces in shadow or strong backlight are
  likely worse. Untested.

### 7. Aspect ratio

- **[community]** Portrait (9:16) works better for people. A standing
  person fills more of the frame height, so their head is bigger. Use it
  for people-centric content when the platform allows.
- 16:9 is fine as long as rules 1–5 are followed.

## Prompt enhancer checklist (at merge)

The enhancer should:

1. Rewrite any shot with a visible person to chest-up or waist-up framing,
   using explicit body cut-offs (rule 1).
2. Strip wide-framing words from shots with visible faces, or turn those
   people away from camera (rules 2–3).
3. Rewrite pull-outs on people to end on a medium shot, or change them to
   push-ins or static shots (rule 4).
4. Cap visible groups at 3, pulled in tight (rule 5).
5. Rewrite crowd shots (rule 5a): sharp foreground people with an
   out-of-focus crowd, or a crowd turned away, in silhouette, or shown
   through signs and hands.
6. Add focus and lighting cues for the main faces (rule 6).

As a safety net, add a simple server-side check that flags a shot whose
text contains both a person word ("man", "woman", "family", "builder",
"people" …) and a wide word (rule 2's list). Use it to warn the user or to
re-run the enhancer. Also flag a shot with a crowd word ("crowd",
"protesters", "audience", "team", "fans", "people gathered" …) that doesn't
say the crowd is out of focus, turned away or in silhouette.

## Test log

| Date | Clip | Shot | Framing as prompted | Result |
|---|---|---|---|---|
| 2026-10-02 | Alpha Timber 15s (original), several runs (base, Realism LoRA, DaSiWa V3, Sage on/off) | 1 | Wide warehouse, no foreground faces | OK |
| | | 2 | Close-up of one tradie at chest height, trucking | **Good** |
| | | 3 | Medium-wide, builder with thumbs-up, truck behind | Face melted |
| | | 4 | Medium shot of a family of 3 stepping out, pulling out to a static wide | All 3 faces melted |
| 2026-10-02 | Family 7s (Shot 4 on its own) | 1 | Same family shot over 7s | Faces bad from the first frame (rules out clip length) |
| 2026-10-03 | Alpha Timber 7s, 3 shots, **koboldcpp live site** (DaSiWa V1 Q5 GGUF, 1280x736, 20 steps, Euler, hi-res on; `ad4ddca3`) | 2 | Medium-wide, builder with thumbs-up | Face melted |
| | | 3 | Medium shot of a family of 3 stepping out, small pull-out to a static wide | Faces melted |
| pending | Alpha Timber 15s (reframed) | 3 | Builder chest-up, truck out of focus behind | ? |
| | | 4 | Family of 3 waist-up, child in dad's arms, small pull-back ending on a medium shot | ? |
| to test | Crowd/protest shot | - | Sharp foreground speaker chest-up, out-of-focus crowd behind (rule 5a) | ? |

Update this table after every framing test, and promote rules from
[hypothesis] to [tested] (or drop them) based on what we see.
