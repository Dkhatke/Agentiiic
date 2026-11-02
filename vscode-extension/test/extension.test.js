const test = require('node:test');
const assert = require('node:assert');
const lib = require('../lib');
const { activate } = require('../extension');

test('shellQuote survives quotes and shell metacharacters', () => {
  assert.strictEqual(lib.shellQuote("it's $(rm -rf ~)", 'linux'), `'it'\\''s $(rm -rf ~)'`);
  assert.strictEqual(lib.shellQuote("it's", 'win32'), "'it''s'");
});

test('quoteProgram leaves bare commands and quotes paths with spaces', () => {
  assert.strictEqual(lib.quoteProgram('agentic', 'linux'), 'agentic');
  assert.strictEqual(lib.quoteProgram('/Users/me/Agentic Project/.venv/bin/agentic', 'linux'), `'/Users/me/Agentic Project/.venv/bin/agentic'`);
});

test('buildRunCommand', () => {
  const c = lib.buildRunCommand({ program: 'agentic', task: 'fix "it"; ls', profile: 'builder', workspace: '/w s', platform: 'linux' });
  assert.strictEqual(c, `agentic run 'fix "it"; ls' --profile builder --workspace '/w s'`);
});

test('selectionTask', () => {
  assert.strictEqual(lib.selectionTask({ task: 'explain', file: 'a.py', startLine: 3, endLine: 3 }), 'About a.py, line 3: explain');
  assert.strictEqual(lib.selectionTask({ task: 'explain', file: 'a.py', startLine: 3, endLine: 9 }), 'About a.py, lines 3-9: explain');
});

function fakeVscode(inputs, selection) {
  const commands = {}, sent = [], terminals = [];
  const vscode = {
    commands: { registerCommand: (id, fn) => { commands[id] = fn; return { dispose() {} }; } },
    workspace: {
      workspaceFolders: [{ uri: { fsPath: '/proj' } }],
      getConfiguration: () => ({ get: (k) => ({ command: 'agentic', port: 8765 })[k] }),
      asRelativePath: () => 'src/app.py',
    },
    window: {
      terminals,
      createTerminal: (o) => { const t = { name: o.name, exitStatus: undefined, show() {}, sendText: (s) => sent.push(s) }; terminals.push(t); return t; },
      showInputBox: async () => inputs.shift(),
      showWarningMessage() {}, showInformationMessage() {},
      activeTextEditor: selection && { selection, document: { uri: {} } },
    },
    env: { openExternal() {} }, Uri: { parse: (s) => s },
  };
  return { vscode, commands, sent };
}

test('registers all commands and sends tasks to one reused terminal', async () => {
  const ctx = { subscriptions: [] };
  const f = fakeVscode(['add tests', 'compare ocr libs']);
  activate(ctx, f.vscode);
  assert.deepStrictEqual(Object.keys(f.commands).sort(),
    ['agentic.openUI', 'agentic.research', 'agentic.run', 'agentic.runOnSelection', 'agentic.securityAudit']);
  await f.commands['agentic.run']();
  await f.commands['agentic.research']();
  f.commands['agentic.securityAudit']();
  assert.strictEqual(f.sent[0], `agentic run 'add tests' --profile auto --workspace '/proj'`);
  assert.match(f.sent[1], /--profile researcher/);
  assert.match(f.sent[2], /--profile security/);
  assert.strictEqual(f.vscode.window.terminals.length, 1);
});

test('selection command includes file and lines, and refuses an empty selection', async () => {
  const ctx = { subscriptions: [] };
  const f = fakeVscode(['find bugs'], { isEmpty: false, start: { line: 4 }, end: { line: 8 } });
  activate(ctx, f.vscode);
  await f.commands['agentic.runOnSelection']();
  assert.match(f.sent[0], /About src\/app\.py, lines 5-9: find bugs/);
  const g = fakeVscode(['x'], { isEmpty: true });
  activate({ subscriptions: [] }, g.vscode);
  await g.commands['agentic.runOnSelection']();
  assert.strictEqual(g.sent.length, 0);
});

test('a cancelled prompt runs nothing', async () => {
  const f = fakeVscode([undefined]);
  activate({ subscriptions: [] }, f.vscode);
  await f.commands['agentic.run']();
  assert.strictEqual(f.sent.length, 0);
});
