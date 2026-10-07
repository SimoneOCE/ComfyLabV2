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
| 2026-10-07 | Polka-dot bag UGC ad, 15s 9:16, 1 reference picture, 5 shots (live site, `53ea2eb1`) | 1, 4, 5 | Selfie-style chest-up and waist-up creator, face large in frame, talking to camera | **Good**: clean faces, natural lip-sync |
| | | 2, 3 | Hands-only close-ups of the bag (touching the button, packing items) | **Good**: bag faithful to the picture in every shot |

Update this table after every framing test, and promote rules from
[hypothesis] to [tested] (or drop them) based on what we see.

## Ad prompts (reference picture or text)

Rules for ad-style videos, written for the site's Prompt Assistant
(`ENHANCE_AD_RULES` and `enhanceAspectRule` in the website's `server.js`).
Users often type only "an ad for this bag", and the assistant can't see the
pictures, so these rules carry the craft. From two hand-written ad prompts on
2026-10-07: a UGC bag ad (tested, see the log) and a trendy music-only edit
(not yet rendered).

### A1. Pick one format; the user's request always wins [hypothesis]

Professional, cinematic and high-production ads stay fully available. The
defaults only apply when the user doesn't say.

- **Aesthetic UGC, no talking** (default for "an ad" in 9:16):
  - Genuine-looking handheld phone footage in a beautiful real place.
  - "Pack with me" style: everyday essentials into a bag, details, heading
    out. Never unboxing or packaging the picture doesn't show.
  - Natural light, gentle sway, the phone adjusting exposure, imperfect
    framing.
  - Soft trending music plus small real sounds.
  - Closed-lip smile on the one face shot.
  - Why it's the default: the user's own direction on 2026-10-07. The
    music-only "trendy edit" read as too high-production; they wanted
    something that looks user-made but aesthetic.
- **Talking UGC**: selfie-style creator talking to camera (asked for with
  review, testimonial, talking, voiceover). **[tested]** 2026-10-07: came
  out convincing.
- **Trendy edit**: high-energy beat-synced cuts, whip pans, orbits, music
  only (asked for with edit, transitions, hype).
- **Polished commercial**: professional studio and lifestyle
  cinematography, dolly/slider/crane moves, controlled light, at most one
  tagline. Default for "an ad" in 16:9 / 1:1, and whenever the user says
  professional, cinematic, high-end or brand film.

### A1b. Make the place specific [hypothesis]

When the user names a vibe ("luxury apartment"), name two or three concrete
details (white marble island, cream bouclé sofa, floor-to-ceiling window
with sheer curtains), and keep one to three props per shot.

### A2. The product is the hero [tested]

- It's in every shot, named the same way each time ("the bag from
  <Picture 1>").
- Include at least one hands-only close-up of its details.
- End on a hero shot of the product.
- Hands-only product shots also keep face counts down (rules 1-3).

### A3. Don't invent the product [hypothesis]

- Only show what the picture shows: no inside, extra colours, logos or
  packaging.
- Handle it plausibly (hold, wear, set down, turn, put everyday items in).
- The assistant can't see the picture, so it never guesses the product's
  colours, materials or details. It names the product the way the user
  does and says "exactly as it appears in <Picture N>".

### A4. No on-screen text [hypothesis]

No captions, prices, logos or titles in the video. H3 doesn't render text
reliably; describe the action instead.

### A5. Speech fits the shot [tested]

- At most one line per shot, about 2.5 words per second of that shot.
- The line goes in quotes and says who speaks, on or off camera.
- 2026-10-07: 3-second shots with 7-10 word lines landed in time.
- Never invent claims (price, materials, awards).

### A6. Music-only edits [hypothesis]

- Write no dialogue at all, and say the music is the only soundtrack.
- Give the music's style and BPM.
- Cuts land on beats, with a bass drop on the first cut.
- Name one sound effect per shot (a whoosh on a whip pan, a thud when the
  product lands).

### A7. Compose for the frame [hypothesis]

- 9:16: one centred subject, people chest-up or waist-up filling the frame,
  products close and large, push-ins and gentle handheld moves.
- The assistant now receives the Generator's aspect.

## Motion swap prompts (Ref2VA: reference video + picture)

Rules for the motion swap / character swap (a reference video `<Video 1>`, one
picture per new person `<Picture N>`, the video's soundtrack `<Audio 1>`).
Learned on the worker, Ref2VA with the 4-step and 8-step turbo, 2026-10-05/06
(rap clips, Albanese/Hanson, IShowSpeed in Thriller). Every free-form attempt
at the Thriller swap came back as the original performer; the official format
plus a full-body swap worked first time.

### S1. Use MiniMax's official full-reference format [tested]

Ref2VA was trained on prompts in MiniMax's six-section format
(MiniMaxAI/MiniMax-H3 `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`). Free-form
paragraphs ("use <Video 1> as the guide... match the exact movements and
framing") made it copy <Video 1> wholesale. The enhancer should always output:

- `subject_definitions:` - `<Video 1> is the source video for the target
  video edit.`; one `<Subject N>` line per person, pointing at their picture;
  `<Audio 1> is the synchronized audio track of <Video 1> and is reused in the
  target video.`
- `summary:` - starts `[video editing + reference generation + audio reuse]
  The target video is an edited version of <Video 1>.`
- `retention_analysis:` - one line per label with the fixed markers:
  `fully_preserved`, `partially_preserved`, `attribute_transfer`,
  `weak_reference`; audio `fully_copy`.
- `detailed_description:` - one style sentence, then `[Shot 1] ...`, later
  shots `[Shot N] At MM:SS.mmm, ...`, citing the labels where they apply.
- `overall_soundscape:` and `non_diegetic_music:`.

### S2. Swap the whole person, not just the face [tested]

Take only the choreography, position and camera move from the video. The new
person keeps their own face, hair, skin and clothes from their picture
(`fully_preserved`), and the video's performer is "replaced in full".
Face-only swaps that keep the original's outfit or hair let the original win
(Thriller came back as MJ; Hanson kept the rapper's hands).

### S3. Don't describe other people's looks [tested]

Background people are only "kept exactly as they appear in <Video 1>".
Describing the zombies' decaying skin and makeup next to the lead painted it
onto the swapped person.

### S4. Let the picture carry the identity [tested]

Don't re-describe what the picture already shows (hair). Add only what
fights the original: the person's skin tone when the original performer has
makeup or a very different skin tone, and the clothes visible in the
picture. Refer to the person as "the young man from <Picture 1>" / their
`<Subject N>` label, tied to the picture every time.

### S5. Positive wording only [tested]

Describe what should be there. "No sunglasses", "low quality", "grainy",
"no new speech or sounds" pulled in the very thing named (the last one froze
every mouth in a rap swap).

### S6. Singing, rapping, lip-sync [tested]

Say who performs and tie it to `<Audio 1>` ("Only <Subject 2> raps the
lyrics of <Audio 1>"); give the non-performer "mouth closed" in the shot
lines. "Rapping" after two names made both rap.

### S7. Restate the swapped people in every shot line [tested]

On long multi-shot clips, people named only once reverted to the originals
near the end.

### Settings that go with it [tested]

- 24fps reference, duration at most 15.0s (H3 rounds frames up; 15.17s
  became 379 frames, past its 362-frame training range).
- Output the same shape as the reference (640x480 source -> 1024x768).
- Ref2VA 4-step turbo is enough when the prompt is right (~4.3 min for 15s
  at 1024x768); 8-step turbo ~7.7 min; no turbo ~26 min.
- One clear, front-facing, well-lit picture per person is enough.

### Template (the Thriller prompt that worked)

```
subject_definitions:
<Video 1> is the source video for the target video edit.
<Subject 1> is the young man in <Picture 1>, exactly as he appears in <Picture 1>: his own face, his own hair, his smooth dark brown skin, and his own outfit from <Picture 1>, the black leather jacket over a black T-shirt with a diamond chain and pendant. He replaces the whole lead dancer of <Video 1> and performs that dancer's choreography in that dancer's position.
<Subject 2> is the group of background dancers in <Video 1>, kept exactly as they appear in <Video 1>.
<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.

summary:
[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. The lead dancer of <Video 1> is replaced in full by <Subject 1>, the young man from <Picture 1>, with his own appearance and outfit, and the original street is replaced by a rain-soaked neon city street at night. <Subject 1> leads <Subject 2> through the same dance, with the same camera move and timing as <Video 1>, while <Audio 1> is reused as the soundtrack.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the young man from <Picture 1> keeps his face, hair, smooth dark brown skin, outfit and full identity, and takes the lead dancer's place and choreography from <Video 1>.
<Subject 2> (appears in [Shot 1]): fully_preserved - the background dancers, their positions and their synchronized moves are kept as in <Video 1>.
<Video 1> (choreography, camera movement and timing): partially_preserved - the dance, the camera pull-back and the timing are kept; the lead dancer is replaced in full by <Subject 1> and the street environment is replaced.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
The target video is a sharp, modern, photorealistic high-definition music video with moody cinematic night lighting and saturated neon colour.
[Shot 1] The shot opens on a medium close-up of <Subject 1>, the young man from <Picture 1>, looking exactly as he does in <Picture 1>, with his smooth dark brown skin, wearing his black leather jacket over a black T-shirt with his diamond chain. He walks toward the camera down a rain-soaked neon city street at night, pink and blue neon signs glowing behind him and reflecting on the wet asphalt, with <Subject 2> following behind him as in <Video 1>. As in <Video 1>, the camera slowly pulls back into a wide shot, revealing <Subject 1> at the front of the formation, leading <Subject 2> through the dance step for step: sharp shoulder isolations, claw-hand gestures, side steps and hip thrusts, every move on the beat of <Audio 1>. Steam rises from street vents and light fog hangs at ground level. <Subject 1> looks exactly like the young man in <Picture 1> from the first frame to the last.

overall_soundscape:
Footsteps on wet asphalt and the city ambience within <Audio 1> continue throughout.

non_diegetic_music:
<Audio 1> is directly reused as the complete soundtrack.
```

For the enhancer: the user should only have to say who replaces whom and
what changes ("swap the lead for Picture 1, new background: neon street");
the enhancer writes the six sections, a shot list from the reference video's
cuts, and applies S2-S7.
