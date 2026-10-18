// P14-0 guard (decision PF-31): Windows long paths are disabled on the dev machine
// (LongPathsEnabled = 0), so every path under node_modules must stay below MAX_PATH (260).
// Node itself copes with longer paths; Python, git and Explorer do not. Run after any
// dependency change: `npm run check-paths`.
import fs from "node:fs";
import path from "node:path";

const LIMIT = 260;
const MARGIN = 20;
let longest = "";
(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (p.length > longest.length) longest = p;
    if (e.isDirectory() && !e.isSymbolicLink()) walk(p);
  }
})(path.resolve("node_modules"));

console.log(`longest path: ${longest.length} chars (limit ${LIMIT}, margin ${MARGIN})`);
console.log(longest);
// MAX_PATH is a Windows limit. On Linux (PATH_MAX 4096) the absolute path is dominated by
// where the checkout lives -- a CI runner nests the repo name twice -- so only report there.
if (process.platform !== "win32") {
  console.log(`not Windows (${process.platform}): MAX_PATH does not apply, reported only`);
  process.exit(0);
}
if (longest.length > LIMIT - MARGIN) {
  console.error("FAIL: within the margin of MAX_PATH -- enable long paths (P14-0) first");
  process.exit(1);
}
