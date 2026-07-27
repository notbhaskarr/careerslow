#!/usr/bin/env node
/**
 * Writes frontend/config.js for Vercel (or local) using API_BASE_URL.
 * Empty string = same origin (Render all-in-one deploy).
 */
const fs = require("fs");
const path = require("path");

const base = (process.env.API_BASE_URL || "").replace(/\/$/, "");
const out = path.join(__dirname, "..", "frontend", "config.js");
const content = `// Auto-generated — do not edit on Vercel deploys
window.API_BASE = ${JSON.stringify(base)};
`;

fs.writeFileSync(out, content, "utf8");
console.log(`Wrote ${out} with API_BASE=${JSON.stringify(base)}`);
