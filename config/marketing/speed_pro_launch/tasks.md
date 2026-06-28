# Campaign Tasks — Speed-Pro Shoe Launch

Master Supervisor work breakdown. Tasks are sequential — each downstream node
depends on the artifact produced upstream. This node (`mcp_visual_factory`)
owns Task 3 only. The other three tasks dispatch to specialist nodes outside
this codebase and are listed here for traceability.

The dispatcher parses every task block, logs each one to the audit trail, and
forwards only the blocks where `assigned_to == mcp_visual_factory` to the
Director.

---

## Task 1: Audience and Cultural Insight Pass

- **assigned_to:** agent_market_researcher
- **action:** Profile the Speed-Pro launch audience (urban distance runners, 24-38, training for a sub-1:40 half-marathon). Surface three culture moments worth riding (rise of solo run-clubs, return of all-weather training, asphalt-as-arena aesthetic). Return a 1-page insight memo plus three competitive launch decks.
- **deliverable:** `research_memo.md`
- **upstream:** none
- **downstream:** agent_copywriter, mcp_visual_factory
- **status:** handled upstream — out of scope for this node

---

## Task 2: Headline and Narrative Copy

- **assigned_to:** agent_copywriter
- **action:** Write the launch headline, deck of three subhead options, and the single brand-promise sentence that will sit beside the hero asset. Tone reference: pared-back performance brand (no exclamation marks, no hype words). The hero sentence must be eight words or fewer.
- **deliverable:** `copy_deck.md`
- **upstream:** Task 1 — `research_memo.md`
- **downstream:** mcp_visual_factory (for tone calibration), agent_legal
- **status:** handled upstream — out of scope for this node

---

## Task 3: Hero Visual Asset — Suspended Shoe Over Wet Asphalt

- **assigned_to:** mcp_visual_factory
- **action:** Render a single hero product shot of the Speed-Pro running shoe suspended mid-frame, two inches above a freshly rain-soaked asphalt track. The asphalt reads as a half-mirror — the shoe's silhouette and brand accent color reflect cleanly downward, distorted slightly by surface texture. Lane markings curve out of focus behind the subject. A thin spray of suspended water droplets traces the trajectory the shoe was just on. Light reads as a 5600K key from camera-right at 35 degrees above horizon, with a soft fill from camera-left to keep the midsole readable. The mood is the millisecond between push-off and landing — frozen, intentional, quiet. The composition is rule-of-thirds left, with the shoe occupying the upper-left third and the reflection anchoring the lower-right diagonal. Aspect ratio 16:9. No humans, no logos, no text inside the frame.
- **style_reference:** `rules.md` — bind to the full guardrail set; the manifest must show every constraint resolved.
- **deliverable:** `hero_asset.png` + `manifest.md`
- **upstream:** Task 1 (insight memo), Task 2 (tonal anchor)
- **downstream:** agent_legal, retoucher hand-off, OOH adaptation
- **status:** **DISPATCH TO THIS NODE**

---

## Task 4: Compliance and Claims Review

- **assigned_to:** agent_legal
- **action:** Audit the rendered asset and accompanying copy against advertising-standards guidance for performance claims, athlete likeness, and trademark adjacency. Flag any incidental detail in the render — a half-visible swoosh-shaped highlight, an asphalt mark that resembles a competitor's wordmark — and return either a clean approval or a structured re-render request.
- **deliverable:** `legal_review.md`
- **upstream:** Task 3 — `hero_asset.png` + `manifest.md`
- **downstream:** final approval gate before media planning
- **status:** handled downstream — out of scope for this node
