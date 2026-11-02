// Agentic Launcher: starts the local `agentic` CLI in an integrated terminal. It never reads or edits your files itself and
// never auto-approves anything: every permission prompt appears in the terminal, as it does when you run the CLI by hand.
const fs = require('fs');
const lib = require('./lib');

function activate(context, vscode = require('vscode')) {
  const cfg = () => vscode.workspace.getConfiguration('agentic');
  const workspaceDir = () => (vscode.workspace.workspaceFolders || [])[0]?.uri.fsPath;

  function terminal(name) {
    const existing = vscode.window.terminals.find((t) => t.name === name && t.exitStatus === undefined);
    return existing || vscode.window.createTerminal({ name, cwd: workspaceDir() });
  }

  function send(taskText, profile) {
    const ws = workspaceDir();
    if (!ws) { vscode.window.showWarningMessage('Agentic: open a folder first.'); return; }
    const t = terminal('Agentic');
    t.show();
    t.sendText(lib.buildRunCommand({ program: cfg().get('command'), task: taskText, profile, workspace: ws }));
  }

  async function ask(prompt, placeHolder) {
    const value = await vscode.window.showInputBox({ prompt, placeHolder, ignoreFocusOut: true });
    return value && value.trim() ? value.trim() : undefined;
  }

  const reg = (id, fn) => context.subscriptions.push(vscode.commands.registerCommand(id, fn));

  reg('agentic.run', async () => {
    const task = await ask('What should the agent do?', 'e.g. add input validation to the signup form');
    if (task) send(task, 'auto');
  });

  reg('agentic.runOnSelection', async () => {
    const ed = vscode.window.activeTextEditor;
    if (!ed || ed.selection.isEmpty) { vscode.window.showInformationMessage('Agentic: select some code first.'); return; }
    const task = await ask('What should the agent do with the selection?', 'e.g. explain this and find bugs');
    if (!task) return;
    const rel = vscode.workspace.asRelativePath(ed.document.uri);
    send(lib.selectionTask({ task, file: rel, startLine: ed.selection.start.line + 1, endLine: ed.selection.end.line + 1 }), 'auto');
  });

  reg('agentic.securityAudit', () => send('Audit this project for security problems and report your findings.', 'security'));

  reg('agentic.research', async () => {
    const q = await ask('What should the agent research?', 'e.g. compare free OCR libraries for Python');
    if (q) send(q, 'researcher');
  });

  reg('agentic.openUI', async () => {
    const port = cfg().get('port') || 8765;
    const t = terminal('Agentic UI');
    t.sendText(`${lib.quoteProgram(cfg().get('command'))} serve --port ${port}`);
    await new Promise((r) => setTimeout(r, 2000)); // give the daemon time to write its token
    try {
      const token = fs.readFileSync(lib.tokenFile(), 'utf8').trim();
      vscode.env.openExternal(vscode.Uri.parse(lib.uiUrl(port, token)));
    } catch {
      vscode.window.showWarningMessage('Agentic: could not read the UI token; open the URL printed in the "Agentic UI" terminal.');
    }
  });
}

function deactivate() {}

module.exports = { activate, deactivate };
