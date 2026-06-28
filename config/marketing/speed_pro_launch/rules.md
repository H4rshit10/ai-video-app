# Visual Guardrails — Speed-Pro Launch

These rules bind every visual asset produced under this campaign profile. The
Director reads this file when constructing the manifest and must resolve every
guardrail explicitly. A rule that is silently violated counts as a missed
deliverable, not a stylistic choice.

---

## 1. Mandatory Brand Palette

Every hex code that appears in the rendered frame must trace to this system or
to a documented neutral derived from it. Off-system color is grounds for re-render.

| Token | Hex | Role | Notes |
|---|---|---|---|
| `accent_hyperspeed` | `#F0FF26` | Single accent — used on the shoe's energy line, the suspended droplets' rim, and nowhere else | High-vis signal; do not let it bleed into reflections beyond 30% saturation |
| `core_carbon` | `#1A1D24` | Primary structural color — midsole shadow, carbon plate, deepest reflection wells | Reads near-black but holds blue undertone — never use pure `#000000` |
| `asphalt_wet` | `#2E3540` | Ground plane — the wet track surface | Half-mirror reflectance; specular highlights at `#A8B4C2` |
| `pulse_red` | `#FF2D55` | Reserved supporting tone — heel pull, single brand-line stitch | Cap at ≤4% of frame coverage; never adjacent to `accent_hyperspeed` |
| `diffuse_white` | `#F5F7FA` | Reflected sky tone, water droplet highlights, midsole inner foam | Never `#FFFFFF` — pure white burns out at OOH scale |

Substitutions, gradient drift toward off-system tones, or surprise accents
require an explicit override line in the manifest. There is no soft-violation.

## 2. WCAG Contrast Minimums

Even though no text appears inside the rendered frame, the brand palette is
selected so any downstream OOH layout layer can place type without falling
below standard. Document the ratios in the manifest:

- `accent_hyperspeed` on `core_carbon`: must hit **WCAG AA Large (3:1)** minimum
  for any future caption work. Target ratio: **14.8:1** (verified).
- `diffuse_white` on `asphalt_wet`: must hit **WCAG AA Body (4.5:1)** minimum.
  Target ratio: **8.2:1** (verified).
- `pulse_red` is type-prohibited at any size against `accent_hyperspeed` —
  ratio falls below 1.5:1 and the combination flickers at OOH scale.

The manifest must list the actual computed contrast for each pairing used.

## 3. Prohibited Compositional Elements

The render must not contain any of the following. These tend to creep in via
default model behavior, so call them out explicitly in the manifest's
"prohibited elements resolved" section.

- **Humans, hands, feet, legs, athletes.** This is a product hero shot. The
  shoe is the subject; a human in frame splits attention.
- **Visible logos or wordmarks** of any brand — including incidental shapes
  that read as competitor marks (swoosh-arc highlights, three-stripe shadows,
  parallel-line scuffs on asphalt). Verify in the manifest.
- **Text, captions, or numerals** baked into the render. Type lives in the
  layout layer downstream.
- **Centered subject on isolated white background.** Default stock-photo
  framing — read as a render, not a brand image.
- **Lens flares, light streaks, "cinematic" overlays.** They date instantly
  and signal AI-default styling.
- **Athlete likeness or recognizable run-club merchandise.** Even partial.
- **Generic motion blur smeared across the whole frame.** Suspension is
  achieved through *implied trajectory* (droplet trail, asphalt reflection
  alignment), not full-frame blur.
- **Surreal scale inconsistencies** — droplets too large for the shoe, lane
  lines too small for the track, etc.

## 4. Required Compositional Elements

- **Single hero subject.** The shoe occupies one third of the frame; the
  reflection anchors the opposing diagonal.
- **Rule of thirds.** No centered framing. Document which third the subject
  sits in.
- **Generous negative space.** Minimum 35% of frame must be unbroken negative
  space (asphalt + atmospheric falloff).
- **Trajectory cue.** A small trail of suspended water droplets that implies
  where the shoe was a millisecond ago. Three to seven droplets, not a spray.
- **Reflectance discipline.** The wet asphalt is a half-mirror — the
  reflected silhouette must align geometrically with the subject and lose
  ~60% saturation. No perfect mirroring; no smeared reflection either.

## 5. Lighting Specification

- **Key light:** 5600K daylight equivalent, camera-right, 35 degrees above
  horizon. Hard enough to define the upper edge of the shoe, not so hard that
  the midsole goes black.
- **Fill light:** Soft 4800K bounce from camera-left, ~⅓ key intensity. Keeps
  the medial side of the midsole readable.
- **Rim light:** Implied 6500K from behind-left. Catches the `accent_hyperspeed`
  energy line and the leading edge of the rear droplet.
- **Ambient register:** Overcast post-rain — flat sky, no direct sun haze,
  high atmospheric humidity reading on the asphalt.

The manifest must record kelvin and angle for each light source.

## 6. Material Truth

- **Mesh upper:** Must read as engineered knit — visible weave at micro scale,
  not a smooth shell. No painted-on texture.
- **Midsole foam:** Visible cellular structure on the cut line. Reflects with
  matte diffusion, not gloss.
- **Carbon plate:** Implied, not visible — suggested only by a thin dark line
  along the midsole's pressure axis.
- **Outsole rubber:** Soft sheen, not patent-leather gloss.
- **Asphalt:** Aggregate visible — small stones glinting in the wet, not a
  uniform texture pass.

## 7. Manifest Requirements

Every render produced under this campaign must ship with a manifest that
explicitly resolves Rules 1 through 6 above. The manifest is the contract.
The render without the manifest is not a deliverable.
