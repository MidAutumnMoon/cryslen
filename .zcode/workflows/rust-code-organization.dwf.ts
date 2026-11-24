/* zcode-workflow
description: 'Brings every Rust file in line with one section of the Rust style
  guide ("Code organization" by default): imports move into the scope that uses
  them, functions are called through their module, types are written without
  their module path. Only use lines and module paths may change; rustfmt, clippy
  and the tests gate the result.'
whenToUse: "When the Rust code needs the Code organization rules of the style
  guide enforced and nothing else: no new type, no changed signature, no renamed
  item. It edits the working copy directly, so run it inside a change you can
  review with `jj diff`. A file the working copy already changed goes straight
  to review, so it also picks up an interrupted pass."
args:
  guide:
    type: string
    description: Path of the Rust style guide file. The run reads it and enforces
      one section of it.
    default: /home/teapot/.agents/skills/language-style-guide/rust.md
  sectionStart:
    type: string
    description: Heading that opens the section of the guide to enforce.
    default: "## Code organization"
  sectionEnd:
    type: string
    description: Heading that follows that section and ends it.
    default: "## Ownership and borrowing"
  localizeMaxRefs:
    type: number
    description: An import used in several functions is imported inside each of them
      while it has at most this many uses in the file; with more uses it stays
      where it is.
    default: 2
  globs:
    type: json
    description: Workspace-relative globs of the Rust files to bring in line.
    default:
      - src/**/*.rs
      - tests/**/*.rs
  package:
    type: string
    description: Cargo package that rustfmt, clippy and the tests run on.
    default: cryslen
  base:
    type: string
    description: Git revision holding the content before the pass. Every file is
      compared with it, and a file that changed beyond use lines and module
      paths is restored from it. In a jj working copy HEAD is the parent commit.
    default: HEAD
*/
type Rule =
  | "avoid-glob-imports"
  | "one-use-per-module"
  | "prefer-locally-scoped-imports"
  | "qualify-top-level-functions"
  | "unqualified-type-names";

// ---- what a subagent hands back ----

interface Move {
  /** Id of the candidate this answers, such as "import:Child". */
  lead: string;
  /** The function or block the use statement now sits in. */
  scope: string;
}
interface Fix {
  /** Id of the candidate this answers; "" when reading the file found it. */
  lead: string;
  /** The function or item that changed. */
  where: string;
  rule: Rule;
  /** One sentence: what changed. */
  what: string;
}
interface Kept {
  /** Id of the candidate. */
  lead: string;
  /** Why it already conforms to the guide. */
  reason: string;
}
interface Deferred {
  lead: string;
  where: string;
  rule: Rule;
  /** One sentence: what violates the guide. */
  what: string;
  /** The change that fixes it. */
  fix: string;
  /** Other files the fix must edit. */
  touches: string[];
}
interface StylistResult {
  /** Imports moved from the top of the file or of a nested module into the function or block that uses them. */
  moved: Move[];
  /** Every other edit made for the guide. */
  fixes: Fix[];
  /** Candidates left as they were. */
  kept: Kept[];
  /** Violations left alone because the fix must edit other files. */
  deferred: Deferred[];
}
interface ReviewIssue {
  /** Id of the candidate this issue answers; "" when reading the file found it. */
  lead: string;
  /** The function, or "path:line", where the fault is. */
  where: string;
  /** One sentence: the fault. */
  problem: string;
}
interface Review {
  /** True only when none of the four conditions fails. */
  approved: boolean;
  /** Empty when approved. */
  issues: ReviewIssue[];
  /** Candidates that already conform, each with the reason. */
  kept: Kept[];
}

// ---- what the script computes ----

interface ImportLead {
  id: string;
  /** The use statement as written, without the trailing semicolon. */
  stmt: string;
  line: number;
  /** Lines that mention the imported name outside use statements and comments. */
  refs: number[];
}
interface ModuleLead extends ImportLead {
  /** The whole line, as it stands in the file. */
  text: string;
  /** The nested module the import sits in. */
  module: string;
}
interface SiteLead {
  id: string;
  line: number;
  /** The line as written. */
  text: string;
}
interface QualifiedUse {
  /** The type as written, with its module path. */
  path: string;
  lines: number[];
}
interface FileLeads {
  path: string;
  imports: ImportLead[];
  module: ModuleLead[];
  /** Imports whose name is lower-case: a module, a function or a macro. */
  lower: ImportLead[];
  braces: SiteLead[];
  globs: SiteLead[];
  /** Types written with their module path, other than a prelude name such as Result. */
  qualified: QualifiedUse[];
  /** Types named like a prelude item (Result); these stay qualified by the author's choice. */
  prelude: QualifiedUse[];
}
interface Diagnostic {
  file: string;
  line: number;
  /** The diagnostic as printed, from its header to the line before the next one. */
  block: string;
}
interface Gate {
  clean: boolean;
  diagnostics: Diagnostic[];
  formatCode: number;
  lintCode: number;
  tail: string;
}
interface FileOutcome {
  path: string;
  /** What the stylist of this run reported; empty when there is none. */
  result: StylistResult;
  /** Candidates no subagent settled when the run ended. */
  unresolved: string[];
  /** Why the file was left alone or put back; "" when its changes stand. */
  heldBack: string;
  /** True when an earlier pass had already changed the file before this run started. */
  earlierPass: boolean;
  /** Why there is no stylist report; "" when there is one. */
  missing: string;
}
interface FileReview {
  path: string;
  index: number;
  position: number;
  review: Review;
  /** True when the reviewer did not answer. */
  failed: boolean;
}
interface Finding {
  /** Workspace-relative path, with a line when it applies. */
  where: string;
  /** One sentence: what is left. */
  what: string;
  /** What showed it. */
  evidence: string;
  /** "verified" when a deterministic check or a separate subagent confirmed it; otherwise "unconfirmed". */
  status: "verified" | "unconfirmed";
  /** How much it matters. */
  severity: "low" | "medium" | "high";
}
interface WorkflowReport {
  /** Two or three sentences answering what the user asked for. */
  conclusion: string;
  findings: Finding[];
  /** What the run checked and how. */
  verified: string[];
  /** What the run did not look at or could not check, and why. */
  notCovered: string[];
}
interface FinalGates {
  formatCode: number;
  lintCode: number;
  testCode: number;
  lintOutput: string;
  testOutput: string;
}

// ---- knobs ----

/** A run argument that is a non-empty text, else its default. */
const argText = (name: string, fallback: string): string => {
  const value = args[name];
  return typeof value === "string" && value !== "" ? value : fallback;
};
/** A run argument that is a non-negative number, else its default. */
const argCount = (name: string, fallback: number): number => {
  const value = args[name];
  return typeof value === "number" && value >= 0 ? value : fallback;
};
/** A run argument that is a list, each entry as text, else its default. */
const argList = (name: string, fallback: string[]): string[] => {
  const value = args[name];
  return Array.isArray(value) ? value.map((entry) => String(entry)) : fallback;
};

const GUIDE_PATH = argText("guide", "/home/teapot/.agents/skills/language-style-guide/rust.md");
const SECTION_START = argText("sectionStart", "## Code organization");
const SECTION_END = argText("sectionEnd", "## Ownership and borrowing");
/** An import used in several functions is imported in each while it has at most this many uses in the file. */
const LOCALIZE_MAX_REFS = argCount("localizeMaxRefs", 2);
/** Workspace-relative globs of the files to bring in line. */
const GLOBS = argList("globs", ["src/**/*.rs", "tests/**/*.rs"]);
/** The Cargo package the formatter, clippy and the tests run on. */
const PACKAGE = argText("package", "cryslen");
/** The Git revision that holds the content before the pass. */
const BASE = argText("base", "HEAD");
const MAX_COVERAGE_ROUNDS = 2;
const MAX_GATE_ROUNDS = 3;
const MAX_REVIEW_ROUNDS = 2;

// ---- reading the files ----

const TOP_USE = /^(?:pub(?:\([^)]*\))?\s+)?use\s+([^;]+);/;
const ANY_USE = /^(\s*)(?:pub(?:\([^)]*\))?\s+)?use\s+([^;]+);/;
const USE_START = /^\s*(?:pub(?:\([^)]*\))?\s+)?use\s/;
const COMMENT = /^\s*\/\//;
const SKIP = /^\s*(?:\/\/|#\[)/;
const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;
const LOWER_LEAF = /(?:^|::)([a-z][a-z0-9_]*)$/;
const QUALIFIED = /\b[a-z_][a-z0-9_]*(?:::[a-z_][a-z0-9_]*)*::[A-Z][a-z0-9][A-Za-z0-9_]*/g;
const HEADER = /^(?:error|warning)(?:\[[^\]]*\])?: /;
const LOCATION = /^\s*-->\s+(\S+?):(\d+):\d+/m;
const MODULE_PATH = /\b[a-z_][a-z0-9_]*(?:::[a-z_][a-z0-9_]*)*::/g;

function leafName(stmt: string): string {
  const alias = /\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(stmt);
  if (alias) return alias[1];
  const parts = stmt.trim().split("::");
  return parts[parts.length - 1].trim();
}

/** Line index of the block that holds the line at `index`, or -1 when it is not a plain block. */
function blockOpener(lines: string[], index: number, indent: number): number {
  for (let k = index - 1; k >= 0; k--) {
    const text = lines[k];
    if (text.trim() === "" || SKIP.test(text)) continue;
    return text.search(/\S/) < indent && /\{\s*$/.test(text) ? k : -1;
  }
  return -1;
}

function leadsFor(path: string, text: string): FileLeads {
  const lines = text.split("\n");
  const top: { stmt: string; line: number; name: string }[] = [];
  const lower: ImportLead[] = [];
  const module: ModuleLead[] = [];
  const braces: SiteLead[] = [];
  const globs: SiteLead[] = [];
  const qualified = new Map<string, number[]>();
  const prelude = new Map<string, number[]>();
  lines.forEach((line, index) => {
    const use = ANY_USE.exec(line);
    if (use) {
      const indent = use[1].length;
      const stmt = use[2].trim();
      if (stmt.includes("{")) {
        braces.push({ id: `granularity:${index + 1}`, line: index + 1, text: line.trim() });
      } else if (stmt.endsWith("*")) {
        if (stmt !== "super::*") globs.push({ id: `glob:${index + 1}`, line: index + 1, text: line.trim() });
      } else if (indent === 0) {
        top.push({ stmt, line: index + 1, name: leafName(stmt) });
        if (LOWER_LEAF.test(stmt)) lower.push({ id: `lower:${index + 1}`, stmt, line: index + 1, refs: [] });
      } else {
        const opener = blockOpener(lines, index, indent);
        if (opener >= 0 && /^mod\b/.test(lines[opener].trim())) {
          module.push({
            id: `module:${index + 1}`,
            stmt,
            line: index + 1,
            refs: [],
            text: line,
            module: lines[opener].trim().replace(/\s*\{$/, ""),
          });
        }
      }
      return;
    }
    if (COMMENT.test(line)) return;
    for (const match of line.match(QUALIFIED) ?? []) {
      const target = /::Result$/.test(match) ? prelude : qualified;
      const found = target.get(match);
      if (found) found.push(index + 1);
      else target.set(match, [index + 1]);
    }
  });
  const counted = lines.map((line, index) => ({ line, index, skip: ANY_USE.test(line) || COMMENT.test(line) }));
  const refsOf = (name: string): number[] => {
    if (!IDENT.test(name) || name === "_") return [];
    const pattern = new RegExp(`\\b${name}\\b`);
    return counted.filter((entry) => !entry.skip && pattern.test(entry.line)).map((entry) => entry.index + 1);
  };
  const imports: ImportLead[] = [];
  for (const item of top) {
    imports.push({ id: `import:${item.name}`, stmt: item.stmt, line: item.line, refs: refsOf(item.name) });
  }
  for (const lead of module) lead.refs = refsOf(leafName(lead.stmt));
  const byRefs = (a: ImportLead, b: ImportLead): number => a.refs.length - b.refs.length;
  const asUses = (entries: Map<string, number[]>): QualifiedUse[] =>
    [...entries.entries()].map(([typePath, typeLines]) => ({ path: typePath, lines: typeLines }));
  return {
    path,
    imports: imports.sort(byRefs),
    module: module.sort(byRefs),
    lower,
    braces,
    globs,
    qualified: asUses(qualified),
    prelude: asUses(prelude),
  };
}

/** Names imported at the top of the file. */
function topLevelImportNames(text: string): Set<string> {
  const names = new Set<string>();
  for (const line of text.split("\n")) {
    const top = TOP_USE.exec(line);
    if (!top || top[1].includes("{") || top[1].trim().endsWith("*")) continue;
    const name = leafName(top[1]);
    if (IDENT.test(name) && name !== "_") names.add(name);
  }
  return names;
}

function isLocalCandidate(lead: ImportLead): boolean {
  return lead.refs.length <= LOCALIZE_MAX_REFS;
}

/** Ids of every candidate the scan finds in a file. */
function candidateIds(file: FileLeads): string[] {
  return [
    ...file.imports.filter(isLocalCandidate).map((lead) => lead.id),
    ...file.module.filter(isLocalCandidate).map((lead) => lead.id),
    ...file.lower.map((lead) => lead.id),
    ...file.globs.map((lead) => lead.id),
    ...file.braces.map((lead) => lead.id),
    ...file.qualified.map((use) => `qual:${use.path}`),
  ];
}

/** Candidates with no answer, and import candidates that still stand where they did without a reason. */
function unanswered(leads: FileLeads, result: StylistResult, text: string): string[] {
  const answered = new Set<string>([
    ...result.moved.map((entry) => entry.lead),
    ...result.fixes.map((entry) => entry.lead),
    ...result.kept.map((entry) => entry.lead),
    ...result.deferred.map((entry) => entry.lead),
  ]);
  const excused = new Set<string>([
    ...result.kept.map((entry) => entry.lead),
    ...result.deferred.map((entry) => entry.lead),
  ]);
  const open: string[] = [];
  const topNames = topLevelImportNames(text);
  for (const lead of leads.imports) {
    if (!isLocalCandidate(lead)) continue;
    if (!answered.has(lead.id)) open.push(lead.id);
    else if (!excused.has(lead.id) && topNames.has(leafName(lead.stmt))) open.push(lead.id);
  }
  const lines = new Set(text.split("\n"));
  for (const lead of leads.module) {
    if (!isLocalCandidate(lead)) continue;
    if (!answered.has(lead.id)) open.push(lead.id);
    else if (!excused.has(lead.id) && lines.has(lead.text)) open.push(lead.id);
  }
  for (const lead of [...leads.lower, ...leads.braces, ...leads.globs]) {
    if (!answered.has(lead.id)) open.push(lead.id);
  }
  for (const use of leads.qualified) {
    if (!answered.has(`qual:${use.path}`)) open.push(`qual:${use.path}`);
  }
  return open;
}

/** Candidates of the file that the review neither kept nor raised. */
function unreviewed(leads: FileLeads, review: Review): string[] {
  const settled = new Set<string>([
    ...review.kept.map((entry) => entry.lead),
    ...review.issues.map((entry) => entry.lead),
  ]);
  return candidateIds(leads).filter((id) => !settled.has(id));
}

/** The file without use statements, blank lines, whitespace and module paths. */
function skeleton(text: string): string {
  const kept: string[] = [];
  let inUse = false;
  for (const line of text.split("\n")) {
    if (inUse) {
      inUse = !line.includes(";");
      continue;
    }
    if (USE_START.test(line)) {
      inUse = !line.includes(";");
      continue;
    }
    if (line.trim() === "") continue;
    kept.push(line);
  }
  return kept
    .join("\n")
    .replace(MODULE_PATH, "")
    .replace(/\s+/g, "");
}

/** True when the file differs from its committed version only in use lines, whitespace and module paths. */
function syntaxOnly(committed: string, current: string): boolean {
  return committed !== "" && skeleton(committed) === skeleton(current);
}

/** Lines other than use statements and blank lines whose number of occurrences differs between two versions. */
function otherChangedLines(before: string, after: string): string[] {
  const tally = (text: string): Map<string, number> => {
    const seen = new Map<string, number>();
    for (const line of text.split("\n")) {
      const key = line.trim();
      if (key === "" || USE_START.test(line)) continue;
      seen.set(key, (seen.get(key) ?? 0) + 1);
    }
    return seen;
  };
  const oldCount = tally(before);
  const newCount = tally(after);
  const out: string[] = [];
  for (const [key, n] of oldCount) {
    for (let k = newCount.get(key) ?? 0; k < n; k++) out.push(`- ${key}`);
  }
  for (const [key, n] of newCount) {
    for (let k = oldCount.get(key) ?? 0; k < n; k++) out.push(`+ ${key}`);
  }
  return out;
}

/** Splits compiler and clippy output into whole diagnostics and names the file of each. */
function parseDiagnostics(output: string): Diagnostic[] {
  const found: Diagnostic[] = [];
  let block: string[] = [];
  const flush = (): void => {
    const whole = block.join("\n").trimEnd();
    const at = whole === "" ? null : LOCATION.exec(whole);
    if (at) found.push({ file: at[1], line: Number(at[2]), block: whole });
  };
  for (const line of output.split("\n")) {
    if (HEADER.test(line)) {
      flush();
      block = [line];
    } else if (block.length > 0) {
      block.push(line);
    }
  }
  flush();
  return found;
}

function blocksFor(gate: Gate, path: string): string {
  const blocks = gate.diagnostics.filter((entry) => entry.file === path).map((entry) => entry.block);
  return blocks.length > 0
    ? blocks.join("\n\n")
    : `No diagnostic names ${path}. The output ends with:\n${gate.tail.slice(-2_000)}`;
}

/** The files the gate flags, with their index among all files. */
function flagged(outcomes: FileOutcome[], gate: Gate): { path: string; index: number }[] {
  return outcomes
    .map((outcome, index) => ({ path: outcome.path, index }))
    .filter(
      ({ path, index }) =>
        outcomes[index].heldBack === "" && gate.diagnostics.some((entry) => entry.file === path),
    );
}

function cleanLine(text: string): string {
  return text.replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
}

function scoped(lead: ImportLead): string {
  const uses =
    lead.refs.length > 0
      ? `used at line ${lead.refs.join(", ")}`
      : "no line names it: a trait whose methods are called, or an item a macro uses";
  return `- ${lead.id} — use at line ${lead.line}: ${lead.stmt} — ${uses}`;
}

function section(title: string, entries: string[]): string {
  return entries.length > 0 ? `${title}\n${entries.join("\n")}` : `${title}\n- none`;
}

function candidateText(file: FileLeads): string {
  const sites = (leads: SiteLead[]): string[] => leads.map((lead) => `- ${lead.id} — ${lead.text}`);
  return [
    section(
      `Imports at the top of the file with at most ${LOCALIZE_MAX_REFS} uses (id, use, lines that use it)`,
      file.imports.filter(isLocalCandidate).map(scoped),
    ),
    section(
      `Imports at the top of a nested module with at most ${LOCALIZE_MAX_REFS} uses`,
      file.module.filter(isLocalCandidate).map((lead) => `${scoped(lead)} — in ${lead.module}`),
    ),
    section(
      "Imports whose name is lower-case, so a module, a function or a macro",
      file.lower.map((lead) => `- ${lead.id} — line ${lead.line}: ${lead.stmt}`),
    ),
    section(
      "A type written with its module path",
      file.qualified.map((use) => `- qual:${use.path} — line ${use.lines.join(", ")}`),
    ),
    section("An import that brings in a whole module other than super", sites(file.globs)),
    section("An import with braces around several items", sites(file.braces)),
  ].join("\n");
}

/** Formats, then compiles and lints every target. */
async function runGate(): Promise<Gate> {
  const format = await world.run("nix", ["develop", "-c", "cargo", "fmt"], { timeoutMs: 300_000 });
  const lint = await world.run(
    "nix",
    ["develop", "-c", "cargo", "clippy", "--all-features", "--all-targets", "--package", PACKAGE],
    { timeoutMs: 600_000, output: { tail: 1_048_576 } },
  );
  const output = `${lint.stdout}\n${lint.stderr}`;
  const diagnostics = parseDiagnostics(output);
  return {
    clean: format.exitCode === 0 && lint.exitCode === 0 && diagnostics.length === 0,
    diagnostics,
    formatCode: format.exitCode,
    lintCode: lint.exitCode,
    tail: output.slice(-4_000),
  };
}

/** The three checks AGENTS.md names, each once. */
async function runFinalGates(): Promise<FinalGates> {
  const format = await world.run("nix", ["develop", "-c", "cargo", "fmt", "--check"], { timeoutMs: 300_000 });
  const lint = await world.run(
    "nix",
    ["develop", "-c", "cargo", "clippy", "--all-features", "--all-targets", "--package", PACKAGE],
    { timeoutMs: 600_000, output: { tail: 1_048_576 } },
  );
  const tests = await world.run(
    "nix",
    ["develop", "-c", "cargo", "test", "--all-features", "--package", PACKAGE],
    { timeoutMs: 1_800_000, output: { tail: 262_144 } },
  );
  return {
    formatCode: format.exitCode,
    lintCode: lint.exitCode,
    testCode: tests.exitCode,
    lintOutput: `${lint.stdout}\n${lint.stderr}`,
    testOutput: `${tests.stdout}\n${tests.stderr}`,
  };
}

const gatesPass = (run: FinalGates): boolean => run.formatCode === 0 && run.lintCode === 0 && run.testCode === 0;

const NOTHING: StylistResult = { moved: [], fixes: [], kept: [], deferred: [] };

function outcomeNote(outcome: FileOutcome): string {
  const kept = outcome.result.kept.length;
  return `${outcome.result.moved.length} moved, ${outcome.result.fixes.length} fixed, ${kept} kept${
    outcome.unresolved.length > 0 ? `, ${outcome.unresolved.length} unsettled` : ""
  }`;
}

// ---- the run ----

artifact.board("files", {
  title: "Files",
  key: "path",
  status: "stage",
  columns: ["styling", "compiling", "in review", "done", "needs you"],
  cardTitle: "path",
  detail: [{ field: "note" }],
});

phase("Read the committed files, the guide and the earlier pass");
const guideRun = await world.run("cat", [GUIDE_PATH]);
if (guideRun.exitCode !== 0) {
  const missing: WorkflowReport = {
    conclusion: `The style guide could not be read from ${GUIDE_PATH}. Nothing was changed.`,
    findings: [],
    verified: [],
    notCovered: ["every file: the run stopped before the first edit"],
  };
  return missing;
}
const sectionStart = guideRun.stdout.indexOf(SECTION_START);
const sectionEnd = guideRun.stdout.indexOf(SECTION_END);
if (sectionStart < 0 || sectionEnd < sectionStart) {
  const missing: WorkflowReport = {
    conclusion: `The style guide at ${GUIDE_PATH} no longer has the sections "${SECTION_START}" and "${SECTION_END}". Nothing was changed.`,
    findings: [],
    verified: [],
    notCovered: ["every file: the run stopped before the first edit"],
  };
  return missing;
}
const rules = guideRun.stdout.slice(sectionStart, sectionEnd).trim();

const paths = (await Promise.all(GLOBS.map((glob) => files.glob(glob)))).flat();
const readAll = (): Promise<string[]> => Promise.all(paths.map((path) => files.read(path)));
// The base revision is the parent of the jj working copy, so this is the content before any pass.
const committed: string[] = (
  await Promise.all(paths.map((path) => world.run("git", ["show", `${BASE}:${path}`])))
).map((shown) => (shown.exitCode === 0 ? shown.stdout : ""));
// Read by command, not through the recorded view: these are the files as they stand now.
const start = (await Promise.all(paths.map((path) => world.run("cat", [path])))).map((run) => run.stdout);

const RULES = `The "Code organization" rules of the author's Rust style guide, verbatim:
${rules}
How they apply in this repository:
- This is application code: favor simplicity and directness.
- rustfmt and clippy already enforce the mechanical part of these rules (glob imports). Do not audit beyond the candidates listed.
- "Used extensively throughout the file": an import whose uses all sit inside one function or block goes into that function or block, however many uses it has. An import used in several functions is imported inside each of them while it has at most ${LOCALIZE_MAX_REFS} uses in the file; with more uses it stays where it is.
- When the unqualified name would clash with another name in scope (Result is the common case), the item stays written with its module path. Introduce no new alias.
- "use super::*;" inside an inline module (the test modules) is the guide's own exception. Leave it alone.`;

const SYNTAX = `These rules are pure syntax. A file may differ from its committed version only in use lines (added, removed or moved) and in a module path added or removed where an item is used. Nothing else may change: no new type or item, no renamed item or file, no changed signature, parameter, bound or body, no removed clone, no changed comment or attribute. The script compares each file with its committed version and puts back any file that breaks this.`;

const LIMITS = `Limits:
- Edit only the file you are given.
- Do not run cargo, rustfmt, clippy or any build or test command. The script formats and compiles the whole crate once every file is done, and sends you what it flags in yours.
- Do not add or change lint attributes (#[allow], #[expect]) or any config file (Cargo.toml, .clippy.toml, .rustfmt.toml). Do not suppress a warning.
- Match the file's import style: one item per use line; the use lines of a function come first in its body, then a blank line. The formatter wraps at 75 columns.
- jj has an external pager here: pass --no-pager. Never run a jj command that rewrites history or the working copy.
- If a candidate is wrong, or an instruction contradicts the guide, say so in the reason instead of working around it.`;

const STYLIST =
  'You bring one Rust file in line with the "Code organization" rules of an author\'s style guide. You make the smallest correct edits and you report exactly what you changed.';
const REVIEWER =
  'You review a Rust file against the "Code organization" rules of an author\'s style guide with fresh eyes. You never edit a file and you run no build or test command. You approve only when you cannot find a fault.';

const stylists = paths.map((path) => agent(`Style guide for ${path}`, STYLIST));

function firstAsk(path: string, file: FileLeads): string {
  return `Bring ${path} in line with the "Code organization" rules of the author's Rust style guide by editing this file.

${RULES}

${SYNTAX}

${LIMITS}

Candidates the script found in ${path}. Line numbers are of the file as it is now. Every id needs one entry in your result: in moved, fixes, kept or deferred.
${candidateText(file)}

The candidates are a starting point. Read the file from top to bottom once and apply the five rules to what they miss: an import used in one function only, a function called without its module, a type written with its module path.

An import candidate is decided by its use lines. Every use inside one function or block puts the use statement there. Uses in several functions put it in each. An item used where a use statement cannot stand (a struct or enum definition, a const, a trait impl header, a macro) stays where it is and goes to kept with that reason.`;
}

function gateAsk(path: string, blocks: string): string {
  return `After your edits the formatter, the compiler or clippy reports this for ${path}:

${blocks}

Fix ${path}. ${SYNTAX} The limits from before hold: this file only, no build command, no lint attribute. If an import cannot stand where you moved it, put it back and keep every other change.`;
}

function fixAsk(path: string, faults: string): string {
  return `A reviewer found these violations of the "Code organization" rules in ${path}:

${faults}

Fix them by editing ${path}.

${RULES}

${SYNTAX}

${LIMITS}`;
}

const NO_REPORT = "An earlier pass changed this file before this run; its report is not available.";

let outcomes: FileOutcome[] = paths.map((path, index) => {
  const earlierPass = start[index] !== committed[index];
  if (committed[index] === "") {
    return { path, result: NOTHING, unresolved: [], heldBack: "not in the committed tree, so it was left alone", earlierPass: false, missing: NO_REPORT };
  }
  if (earlierPass && !syntaxOnly(committed[index], start[index])) {
    return { path, result: NOTHING, unresolved: [], heldBack: "an earlier pass changed it beyond use lines and module paths, so it was left alone", earlierPass, missing: NO_REPORT };
  }
  return { path, result: NOTHING, unresolved: [], heldBack: "", earlierPass, missing: earlierPass ? NO_REPORT : "" };
});

/** Restores each file to its committed version and marks it as needing the user. */
async function putBack(files: { path: string; index: number }[], why: string): Promise<void> {
  for (const { path, index } of files) {
    const restored = await world.run("jj", ["restore", path]);
    const reason =
      restored.exitCode === 0
        ? `put back to the committed version: ${why}`
        : `${why}, and jj restore exited ${restored.exitCode}`;
    log(`${path} ${reason}`);
    outcomes[index] = { ...outcomes[index], result: NOTHING, unresolved: [], heldBack: reason };
    report({ path, stage: "needs you", note: reason }, "files");
  }
}

/** Files whose change against the committed version is not pure syntax. */
async function scopeBreakers(): Promise<{ path: string; index: number }[]> {
  const now = await readAll();
  return outcomes
    .map((outcome, index) => ({ path: outcome.path, index }))
    .filter(({ index }) => outcomes[index].heldBack === "" && !syntaxOnly(committed[index], now[index]));
}

for (let index = 0; index < paths.length; index++) {
  const outcome = outcomes[index];
  const stage = outcome.heldBack !== "" ? "needs you" : outcome.earlierPass ? "in review" : "styling";
  const note =
    outcome.earlierPass && outcome.heldBack === ""
      ? "changed by the earlier pass"
      : `${candidateIds(leadsFor(outcome.path, start[index])).length} candidates`;
  report({ path: outcome.path, stage, note }, "files");
}
const fresh = outcomes
  .map((outcome, index) => ({ path: outcome.path, index }))
  .filter(({ index }) => !outcomes[index].earlierPass && outcomes[index].heldBack === "");
const earlier = outcomes.filter((outcome) => outcome.earlierPass && outcome.heldBack === "").length;

if (fresh.length > 0) {
  phase("Style each untouched file against the guide");
  log(`${fresh.length} files to style; ${earlier} files the earlier pass changed go straight to review`);
  // One join: the compiler and the formatter below see the whole crate at once.
  const styled = await Promise.all(
    fresh.map(async ({ path, index }) => {
      const stylist = stylists[index];
      try {
        let text = await files.read(path);
        const file = leadsFor(path, text);
        let result = await stylist.ask<StylistResult>(firstAsk(path, file));
        for (let round = 1; round <= MAX_COVERAGE_ROUNDS; round++) {
          const open = unanswered(file, result, text);
          if (open.length === 0) break;
          result = await stylist.ask<StylistResult>(
            `Your result does not settle these candidates of ${path}: ${open.join(", ")}. An import that still stands at the top of the file needs a reason in kept. Send the complete result again, with every candidate answered.`,
          );
          text = await files.read(path);
        }
        text = await files.read(path);
        if (!syntaxOnly(committed[index], text)) {
          await stylist.ask(
            `Your edit to ${path} changed more than use lines and module paths. These lines differ from the committed version beyond a module path:\n${otherChangedLines(
              committed[index],
              text,
            )
              .slice(0, 60)
              .join("\n")}\n\nUndo every change that is not a use line or a module path, and keep the import moves. ${SYNTAX}`,
          );
          text = await files.read(path);
        }
        const outcome: FileOutcome = {
          path,
          result,
          unresolved: unanswered(file, result, text),
          heldBack: "",
          earlierPass: false,
          missing: "",
        };
        if (!syntaxOnly(committed[index], text)) log(`${path}: changed more than use lines and module paths`);
        report({ path, stage: "compiling", note: outcomeNote(outcome) }, "files");
        return { index, outcome };
      } catch (error) {
        const reason = `The stylist did not answer: ${String(error)}`;
        log(`${path}: ${reason}`);
        return { index, outcome: { path, result: NOTHING, unresolved: [], heldBack: "", earlierPass: false, missing: reason } };
      }
    }),
  );
  for (const entry of styled) outcomes[entry.index] = entry.outcome;
}

phase("Format, compile and lint the whole crate");
let gate = await runGate();
for (let round = 1; round <= MAX_GATE_ROUNDS && !gate.clean; round++) {
  const failing = flagged(outcomes, gate);
  if (failing.length === 0) break;
  log(`gate round ${round}: ${failing.length} files flagged`);
  phase("Fix what the compiler and linter flag");
  await Promise.all(
    failing.map(async ({ path, index }) => {
      try {
        await stylists[index].ask(gateAsk(path, blocksFor(gate, path)));
      } catch (error) {
        // The file stays flagged, and is put back below if the gate keeps failing.
        log(`${path}: ${String(error)}`);
      }
    }),
  );
  phase("Format, compile and lint the whole crate");
  gate = await runGate();
}
const stillFailing = flagged(outcomes, gate);
if (stillFailing.length > 0) {
  phase("Put back the files that still fail the build");
  await putBack(stillFailing, "the build still flagged it after three fix rounds");
}
const breakers = await scopeBreakers();
if (breakers.length > 0) {
  phase("Put back the files that changed more than syntax");
  await putBack(breakers, "it changed more than use lines and module paths");
}
if (!gate.clean) {
  phase("Format, compile and lint the whole crate");
  gate = await runGate();
}

let treeClean = gate.clean;
if (!treeClean) {
  treeClean = await hole<boolean>(
    "Decide how to repair the tree after the styling stage",
    `After the styling stage the crate still fails: formatter exit ${gate.formatCode}, clippy exit ${gate.lintCode}. Files put back: ${outcomes
      .filter((outcome) => outcome.heldBack !== "")
      .map((outcome) => outcome.path)
      .join(", ") || "none"}.\n${gate.tail}`,
  );
}

let reviews: FileReview[] = [];
let after = await readAll();
if (treeClean) {
  const changed = outcomes
    .map((outcome, index) => ({ path: outcome.path, index }))
    .filter(({ index }) => outcomes[index].heldBack === "" && after[index] !== committed[index]);
  log(`${changed.length} of ${paths.length} files carry a change`);
  for (const { path, index } of changed) {
    report({ path, stage: "in review", note: `${candidateIds(leadsFor(path, after[index])).length} candidates left to judge` }, "files");
  }

  if (changed.length > 0) {
    const reviewers = changed.map(({ path }) => agent(`Review of ${path}`, REVIEWER));
    const reviewAsk = (path: string, index: number, again: boolean): string => {
      const outcome = outcomes[index];
      const work =
        outcome.missing === "" ? `What the stylist reports:\n${JSON.stringify(outcome.result)}` : outcome.missing;
      const againText = again
        ? "\n\nThe file changed since your last review. Check the whole file again, not only the faults you raised."
        : "";
      return `Review ${path} in the working copy against the "Code organization" rules of the author's Rust style guide. You have not seen this change before. Do not edit any file.

${RULES}

${work}

The script compared the file with its committed version: outside use lines, only whitespace and module paths differ.

Candidates the scan finds in the file now. Each one either already conforms (put it in kept, with the reason) or violates the rules (make it an issue):
${candidateText(leadsFor(path, after[index]))}

See the change with: jj diff --git --no-pager ${path}

Approve only when all four hold. Each one that fails becomes an issue with its location:
1. Every import the rules want local is local: in the narrowest function or block that covers its uses, or in each of several functions as the rule allows. None stands at the top of a nested module when the rule would put it inside a function. None is duplicated where one scope would do.
2. Every kept reason holds against the code.
3. Every module path added or removed follows the rules and the file's import style: a function is called through its module, a type is written without its module path, nothing is aliased.
4. Nothing in ${path} still violates the five rules.${againText}`;
    };
    const faults = (entry: FileReview): string =>
      entry.review.issues.map((issue) => `- ${issue.where}: ${issue.problem}`).join("\n");

    // One join: the fix round below edits files.
    phase("Independent review of each changed file");
    reviews = await Promise.all(
      changed.map(async ({ path, index }, position): Promise<FileReview> => {
        try {
          let review = await reviewers[position].ask<Review>(reviewAsk(path, index, false));
          for (let round = 1; round <= MAX_COVERAGE_ROUNDS; round++) {
            const open = unreviewed(leadsFor(path, after[index]), review);
            if (open.length === 0) break;
            review = await reviewers[position].ask<Review>(
              `Your review does not settle these candidates in ${path}: ${open.join(", ")}. Each one goes to kept with a reason, or to issues. Send the complete review again.`,
            );
          }
          return { path, index, position, review, failed: false };
        } catch (error) {
          log(`${path}: the reviewer did not answer: ${String(error)}`);
          return { path, index, position, review: { approved: false, issues: [], kept: [] }, failed: true };
        }
      }),
    );

    for (let round = 1; round <= MAX_REVIEW_ROUNDS; round++) {
      const open = reviews.filter((entry) => !entry.failed && !entry.review.approved);
      if (open.length === 0) break;
      log(`review round ${round}: ${open.length} files with findings`);

      phase("Address the review findings");
      await Promise.all(
        open.map(async (entry) => {
          try {
            await stylists[entry.index].ask(fixAsk(entry.path, faults(entry)));
          } catch (error) {
            log(`${entry.path}: the stylist did not answer: ${String(error)}`);
          }
        }),
      );

      phase("Format, compile and lint the whole crate");
      gate = await runGate();
      if (!gate.clean) {
        phase("Fix what the compiler and linter flag");
        await Promise.all(
          flagged(outcomes, gate).map(async ({ path, index }) => {
            try {
              await stylists[index].ask(gateAsk(path, blocksFor(gate, path)));
            } catch (error) {
              log(`${path}: ${String(error)}`);
            }
          }),
        );
        phase("Format, compile and lint the whole crate");
        gate = await runGate();
      }
      if (!gate.clean) break;

      const reviewBreakers = await scopeBreakers();
      if (reviewBreakers.length > 0) {
        phase("Put back the files that changed more than syntax");
        await putBack(reviewBreakers, "a review fix changed more than use lines and module paths");
      }

      after = await readAll();
      phase("Check the revised files again");
      const rechecked = await Promise.all(
        open.map(async (entry): Promise<FileReview> => {
          if (outcomes[entry.index].heldBack !== "") return entry;
          try {
            const review = await reviewers[entry.position].ask<Review>(reviewAsk(entry.path, entry.index, true));
            return { ...entry, review, failed: false };
          } catch (error) {
            log(`${entry.path}: the reviewer did not answer: ${String(error)}`);
            return { ...entry, failed: true };
          }
        }),
      );
      reviews = reviews.map((entry) => rechecked.find((candidate) => candidate.position === entry.position) ?? entry);
    }

    for (const entry of reviews) {
      const heldBack = outcomes[entry.index].heldBack;
      const done = heldBack === "" && !entry.failed && entry.review.approved;
      report(
        {
          path: entry.path,
          stage: done ? "done" : "needs you",
          note:
            heldBack !== ""
              ? heldBack
              : done
                ? `${entry.review.kept.length} candidates kept with a reason`
                : `${entry.review.issues.length} review findings open`,
        },
        "files",
      );
    }
  }
}

// A fix round may have edited files; count what is still open.
after = await readAll();
outcomes = outcomes.map((outcome, index): FileOutcome => {
  if (outcome.heldBack !== "" || after[index] === committed[index]) return outcome;
  const review = reviews.find((entry) => entry.index === index);
  const settled = new Set<string>([
    ...outcome.result.moved.map((entry) => entry.lead),
    ...outcome.result.fixes.map((entry) => entry.lead),
    ...outcome.result.kept.map((entry) => entry.lead),
    ...outcome.result.deferred.map((entry) => entry.lead),
    ...(review === undefined ? [] : review.review.kept.map((entry) => entry.lead)),
    ...(review === undefined ? [] : review.review.issues.map((entry) => entry.lead)),
  ]);
  const open = candidateIds(leadsFor(outcome.path, after[index])).filter((id) => !settled.has(id));
  return open.length === 0 ? outcome : { ...outcome, unresolved: open };
});
for (let index = 0; index < paths.length; index++) {
  if (outcomes[index].heldBack === "" && after[index] === committed[index]) {
    report({ path: paths[index], stage: "done", note: "no change" }, "files");
  }
}

phase("Run the formatter, the linter and the full test suite");
let last = await runFinalGates();
let repaired = false;
if (!treeClean || !gatesPass(last)) {
  repaired = await hole<boolean>(
    "Decide how to repair the failing final gate",
    `${
      treeClean ? "The crate built after the styling stage." : "The crate does not build after the styling stage."
    } Formatter exit ${last.formatCode}, clippy exit ${last.lintCode}, tests exit ${last.testCode}.
Clippy output ends with:
${last.lintOutput.slice(-3_000)}
Test output ends with:
${last.testOutput.slice(-3_000)}`,
  );
  if (repaired) {
    treeClean = true;
    phase("Run the formatter, the linter and the full test suite");
    last = await runFinalGates();
  }
}
const passed = treeClean && gatesPass(last);

const keptOf = (index: number): Kept[] => {
  const review = reviews.find((entry) => entry.index === index);
  return review === undefined ? outcomes[index].result.kept : review.review.kept;
};
const approvedCount = reviews.filter((entry) => !entry.failed && entry.review.approved).length;
const changedPaths = paths.filter((_, index) => after[index] !== committed[index]);
const leftBroken = changedPaths.filter((_, index) => !syntaxOnly(committed[index], after[index]));
const preludeTotals = new Map<string, { lines: number; files: number }>();
for (const file of paths.map((path, index) => leadsFor(path, after[index]))) {
  for (const use of file.prelude) {
    const soFar = preludeTotals.get(use.path);
    if (soFar) {
      soFar.lines += use.lines.length;
      soFar.files += 1;
    } else {
      preludeTotals.set(use.path, { lines: use.lines.length, files: 1 });
    }
  }
}
const testSummary = last.testOutput.match(/test result: .*/g)?.join("; ") ?? "no test summary in the output";

const findings: Finding[] = [];
for (let index = 0; index < outcomes.length; index++) {
  const outcome = outcomes[index];
  const review = reviews.find((entry) => entry.index === index);
  const reviewed = review !== undefined && !review.failed && review.review.approved;
  if (outcome.heldBack !== "") {
    findings.push({
      where: outcome.path,
      what: outcome.heldBack,
      evidence: "the gate's output or the script's comparison with the committed version",
      status: "verified",
      severity: "low",
    });
  }
  if (review !== undefined && !reviewed) {
    findings.push({
      where: outcome.path,
      what: "the review is not closed",
      evidence: review.failed
        ? "the reviewer did not answer"
        : review.review.issues.map((issue) => `${issue.where}: ${issue.problem}`).join("; "),
      status: "verified",
      severity: "low",
    });
  }
  for (const id of outcome.unresolved) {
    findings.push({
      where: `${outcome.path}: ${id}`,
      what: "a candidate no subagent settled",
      evidence: "the script listed the file's candidates again after the last edit",
      status: "verified",
      severity: "low",
    });
  }
  for (const entry of outcome.result.deferred) {
    findings.push({
      where: entry.where,
      what: `${entry.rule}: ${entry.what} is still there. The fix: ${entry.fix} (edits ${entry.touches.join(", ") || "no other file"})`,
      evidence: reviewed
        ? "the independent review of the file saw it and approved the file"
        : "reported by the stylist; no independent review saw it",
      status: reviewed ? "verified" : "unconfirmed",
      severity: "low",
    });
  }
}
for (const path of leftBroken) {
  findings.push({
    where: path,
    what: "differs from its committed version beyond use lines and module paths",
    evidence: "the script's comparison after the last edit",
    status: "verified",
    severity: "low",
  });
}
for (const finding of findings) report(finding);

const changedIndices = paths.map((path, index) => index).filter((index) => after[index] !== committed[index]);
const movedTotal = changedIndices.reduce((sum, index) => sum + outcomes[index].result.moved.length, 0);
const fixesTotal = changedIndices.reduce((sum, index) => sum + outcomes[index].result.fixes.length, 0);
const keptTotal = changedIndices.reduce((sum, index) => sum + keptOf(index).length, 0);
const styledHere = fresh.length;

const conclusion =
  changedPaths.length === 0
    ? "No file differs from its committed version."
    : `${changedPaths.length} of ${paths.length} Rust files differ from their committed version, and only in use lines and module paths. This run styled ${styledHere} files (${movedTotal} imports moved, ${fixesTotal} other fixes); ${earlier} files keep the import moves of the earlier pass. ${keptTotal} candidates stay as they are, each with its reason in the report. ${
        passed ? "rustfmt, clippy and the full test suite pass." : "The final gate does not pass."
      }${leftBroken.length > 0 ? ` ${leftBroken.length} files still differ beyond use lines and module paths.` : ""}${
        findings.length > 0 ? ` ${findings.length} items are left for you, listed in the report.` : ""
      }`;

const verified = [
  `cargo fmt --check: exit ${last.formatCode}`,
  `cargo clippy --all-features --all-targets --package ${PACKAGE}: exit ${last.lintCode}, ${parseDiagnostics(last.lintOutput).length} diagnostics`,
  `cargo test --all-features --package ${PACKAGE}: exit ${last.testCode} (${testSummary})`,
  ...(leftBroken.length === 0
    ? ["the script compared every changed file with its committed version: outside use lines, only whitespace and module paths differ"]
    : []),
  `${reviews.length} changed files, each read by a separate reviewer that had not seen the change; ${approvedCount} approved at the end`,
  "every candidate the scan finds was listed again by the script after the last edit",
  ...(repaired ? ["the final gate failed during the run, was repaired by hand, and was run again"] : []),
];

const notCovered = [
  'Every rule of the guide outside "Code organization": ownership and borrowing (Rc, Arc, clones) and the function-parameter rule (flexible references). Out of scope for this pass, as you asked; nothing checked them.',
  "The visual smoke test from AGENTS.md: it drives the real Wayland session, so this unattended run did not start the app.",
  "The base of the guide, the official Rust Style Guide and the Rust API Guidelines: covered only through the repository's rustfmt and clippy configuration.",
  ...[...preludeTotals.entries()].map(
    ([typePath, count]) => `${typePath} stays written with its module path at ${count.lines} lines in ${count.files} files, as you chose`,
  ),
  ...(earlier > 0
    ? ["the import moves of the earlier pass: no stylist report exists for them, so a reviewer judged each from the diff alone"]
    : []),
];
if (!passed) notCovered.push("The final gate: it does not pass.");

const fileRows = paths.map((path, index) => {
  const outcome = outcomes[index];
  const review = reviews.find((entry) => entry.index === index);
  const state =
    outcome.heldBack !== ""
      ? "needs you"
      : after[index] === committed[index]
        ? "no change"
        : review !== undefined && !review.failed && review.review.approved
          ? "reviewed"
          : "needs you";
  return `| ${path} | ${outcome.earlierPass ? "earlier pass" : outcome.heldBack === "" && after[index] !== committed[index] ? "this run" : "none"} | ${candidateIds(leadsFor(path, after[index])).length} | ${outcome.result.moved.length} | ${outcome.result.fixes.length} | ${keptOf(index).length} | ${state} |`;
});
const keptSection = changedPaths.flatMap((path) => {
  const kept = keptOf(paths.indexOf(path));
  return kept.length === 0 ? [] : [`### ${path}`, ...kept.map((entry) => `- ${cleanLine(`${entry.lead} — ${entry.reason}`)}`)];
});
const list = (lines: string[]): string[] => (lines.length === 0 ? ["- none"] : lines.map((line) => `- ${cleanLine(line)}`));

await artifact.markdown(
  "report",
  [
    "# Rust style guide: code organization",
    "",
    conclusion,
    "",
    "## Gates",
    `- cargo fmt --check: exit ${last.formatCode}`,
    `- cargo clippy --all-features --all-targets --package ${PACKAGE}: exit ${last.lintCode}, ${parseDiagnostics(last.lintOutput).length} diagnostics`,
    `- cargo test --all-features --package ${PACKAGE}: exit ${last.testCode} — ${testSummary}`,
    ...(repaired ? ["- The final gate failed during the run, was repaired by hand, and was run again."] : []),
    "",
    "## Files",
    "| File | Styled in | Candidates left | Imports moved | Other fixes | Kept | State |",
    "| --- | --- | --- | --- | --- | --- | --- |",
    ...fileRows,
    "",
    "## Candidates kept as they are",
    ...(keptSection.length === 0 ? ["- none"] : keptSection),
    "",
    "## Left for you",
    ...list(findings.map((finding) => `${finding.where} (${finding.status}): ${finding.what}`)),
    "",
    "## Not covered",
    ...notCovered.map((line) => `- ${line}`),
  ].join("\n"),
  {
    title: "Rust style guide: code organization",
    description: "What changed in each file, what stayed and why, and what the gates said.",
    primary: true,
  },
);

const result: WorkflowReport = { conclusion, findings, verified, notCovered };
return result;
