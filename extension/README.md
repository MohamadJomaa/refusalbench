# Extension

Your team's own novel, useful extension of the baseline in `baseline/`
goes here. This is the part nobody else has published, the part the
idea-pitch issue's related-work check exists to confirm.

This folder's own README (replace this file, or add alongside it) must
state:

- What specifically you changed or added relative to the baseline, in
  enough detail that a reader does not have to diff the two folders to
  find out.
- The specific limitation of the baseline paper this addresses (should
  match the idea-pitch issue; if it drifted, note why).
- How it is evaluated against the baseline, with a real, direct
  comparison, not just a claim that it is better.
- If a smoke test stood in for full-scale validation, what full-scale
  validation would actually require, stated plainly.

Keep this folder's code importing from or calling into `baseline/` rather
than duplicating it. If your extension requires forking a specific file
from the baseline to modify it, copy only that file here and say exactly
which one and why, rather than copying the whole baseline a second time.
