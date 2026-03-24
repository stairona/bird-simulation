# Bird Simulation — Development Rules

This file applies to this folder and everything below it.

## File Safety (Hard Rule)
- No file or folder deletions.
- No destructive commands (`rm`, `git clean`, `git reset --hard`, etc.).
- Allowed: edit, move, rename, copy, and create files.

## End-of-Session Report (Required)
- "Moved/Renamed Files": old path, new path, reason.
- "Deletion Candidates": path, reason, risk level, and suggested command for manual deletion (user-run only).

## Git Safety
- Always verify current repository before staging/committing.
- Never mix changes across repositories.
- Keep commits small, scoped, and clearly named.
- Do not force push without explicit approval.

## Multi-Project Workspace
This project is part of the `/Users/nicolasaguirre/zprojects/` workspace.
- Each project is an independent git repository.
- Always check current directory before running git commands.
- Respect repository boundaries.
