### Do

- Run `ruff format` and `ruff check` before finalizing changes.
- Use `uv` for dependency and task execution (`uv sync`, `uv run run_sim.py`, `uv run pytest`).
- Add or update tests when behavior changes.
- Keep functions small, explicit, and easy to reason about.
- Export intended public symbols in `__all__`.
- Add docstrings to every non-trivial function (public and private) with summary, `Args`, `Returns`, and examples when behavior is non-obvious.
- Maintain `docs/` with up-to-date design notes, architecture decisions, and user-facing documentation.
- Ask the user when work is stuck or behavior diverges from plan; do not hide issues behind silent workarounds.
- Maintain good test coverage for all functions, and ensure the test suite passes after every change.

### Don't

- Do not bypass `uv` for installing or running project tasks.
- Do not swallow errors; fail explicitly and early when a case cannot be handled.
- Do not introduce hidden global state or mutable module-level singletons.
- Do not leave private helpers undocumented if they contain logic.
- Do not change public behavior without tests that cover the new behavior.
- Do not mix unrelated refactors into a focused change.
