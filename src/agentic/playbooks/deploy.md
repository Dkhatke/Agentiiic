# Deploy playbook
Goal: ship the project safely. Never skip a stage; ask the user before each one.

1. **Inspect**: repo_map, read README and build config. Identify the stack and the target (static host, container, PaaS).
2. **Pre-flight**: working tree clean? tests pass? no secrets in the diff (search for keys, `.env` tracked by git)?
   Report anything failing and stop.
3. **Dry run**: show the exact commands and files that would change or be uploaded. Wait for approval.
4. **Build** locally. Show the output size and any warnings.
5. **Release** only after explicit approval of the exact command. Prefer a preview/staging target first.
6. **Verify**: fetch the deployed URL and check it responds. Report the result.
7. **Rollback note**: write down how to undo this deploy (previous version, command).
Never put secrets in commands or files; use the platform's secret store.
