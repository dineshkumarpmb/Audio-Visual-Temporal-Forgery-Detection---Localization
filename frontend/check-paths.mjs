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
if (longest.length > LIMIT - MARGIN) {
  console.error("FAIL: within the margin of MAX_PATH -- enable long paths (P14-0) first");
  process.exit(1);
}
