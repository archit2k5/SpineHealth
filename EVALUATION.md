# Evaluation

## Method
11 frames sampled from a single demo recording, manually labeled for posture (upright/slouch) and compared against the system's on-screen ratio/status output (NPR, `SLOUCH_RATIO_THRESHOLD = 0.6`).

## Result
10/11 (91%) correct.

## Cases

**Transition frame (not a true error):** One frame flagged "bad posture" fell exactly at the moment the person moved from slouching to upright — the status caught the state mid-transition, corrected within the next frame(s). Counted as a pass, not a model error.

**Actual failure — false negative:** System classified a slouch as good posture. Cause: the back was locally straight (high shoulder-to-head vertical extension) but the whole torso was leaning/slanted, which NPR — a vertical-distance-over-shoulder-width ratio — doesn't capture. NPR measures forward head/neck collapse, not whole-body lean angle.

## Known limitation
NPR is blind to whole-torso lean when the neck/back segment itself stays straight. It detects "head dropping toward shoulders," not "body tilting off vertical."
