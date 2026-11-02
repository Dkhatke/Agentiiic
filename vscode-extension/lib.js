// Pure helpers (no vscode API) so they can be tested with plain node.
const os = require('os');
const path = require('path');

function shellQuote(s, platform = process.platform) {
  if (platform === 'win32') return `'${String(s).replace(/'/g, "''")}'`; // PowerShell
  return `'${String(s).replace(/'/g, `'\\''`)}'`;                          // sh, bash, zsh
}

// The program name may contain spaces (an absolute path), so quote it unless it is a bare command.
function quoteProgram(cmd, platform = process.platform) {
  return /^[\w./-]+$/.test(cmd) ? cmd : (platform === 'win32' ? '& ' : '') + shellQuote(cmd, platform);
}

function buildRunCommand({ program, task, profile, workspace, platform }) {
  const parts = [quoteProgram(program, platform), 'run', shellQuote(task, platform)];
  if (profile) parts.push('--profile', profile);
  if (workspace) parts.push('--workspace', shellQuote(workspace, platform));
  return parts.join(' ');
}

function selectionTask({ task, file, startLine, endLine }) {
  const where = startLine === endLine ? `line ${startLine}` : `lines ${startLine}-${endLine}`;
  return `About ${file}, ${where}: ${task}`;
}

function tokenFile(env = process.env) {
  return path.join(env.AGENTIC_STATE || path.join(os.homedir(), '.local', 'state', 'agentic'), 'token');
}

function uiUrl(port, token) {
  return `http://127.0.0.1:${port}/#token=${token}`;
}

module.exports = { shellQuote, quoteProgram, buildRunCommand, selectionTask, tokenFile, uiUrl };
