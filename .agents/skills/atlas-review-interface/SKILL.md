---
name: atlas-review-interface
description: Design or test Atlas photo-review and mapping workflows when contributors inspect obstacle regions, submit conflicting observations, or browse shared evidence.
---

# Design an evidence-first inspection surface

1. Start from the contributor's decision. Show the original scene, an optional overlay, a textual list of regions, and what happens after an observation is submitted. Keep raw votes visible.
2. Put scene context, current review state and primary action before secondary research information. Use one dominant image area. Keep forms usable at a narrow viewport and at 200% zoom. Give every field a persistent accessible name and every status a text label.
3. Separate observed physical features from judgments about passability. Let contributors record an unknown or obscured view. For opposite reports on the same feature, surface both reports and invite a new observation. Provide a list alternative to map-only navigation.
4. Use normalized coordinates relative to the displayed, orientation-corrected photo. Check one portrait and one landscape image for overlay alignment. Display a readable empty state when the model proposes no region. Make synthetic evidence identifiable within the inspection view.
5. Exercise a full path: upload an image, inspect it, submit an observation, refresh, reopen it in the queue and inspect the stored evidence. Test tab order, focus visibility, keyboard submission and recovery from a malformed upload. Record any untested checks.
   
Source: [Vercel web design guidelines skill](https://github.com/vercel-labs/agent-skills/tree/main/skills/web-design-guidelines).
