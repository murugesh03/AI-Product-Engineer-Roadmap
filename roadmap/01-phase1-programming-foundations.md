# Phase 1: Programming Foundations

## 1.1 Python Core Skills
**Why it matters:** AI frameworks and product pipelines are Python-first.
**What it is:** master syntax, OOP, modules, typing, and testing.
**Goal:** write clean, production-grade Python code with tests.

### Steps
1. Learn variables, data structures, loops, and functions.
2. Practice file I/O, JSON/CSV handling.
3. Build reusable modules and packages.
4. Add `pytest` tests and run CI locally.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `CODE_STYLE.md` (if exists)
- `pytest` docs

---

## 1.2 API Development (FastAPI + Flask)
**Why it matters:** product-ready AI services need stable endpoints.
**What it is:** design REST/GraphQL endpoints, request validation, auth.
**Goal:** deploy a minimal API to local docker with health checks.

### Steps
1. Create FastAPI skeleton + Uvicorn entrypoint.
2. Add routes for prediction and status.
3. Add request schema with Pydantic.
4. Write integration tests.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `app/main.py`
- `tests/test_api.py`

---

## 1.3 Code Quality and Workflow
**Why it matters:** maintainers rely on linting and formatting.
**What it is:** pre-commit hooks, linters, git flow.
**Goal:** stable developer experience and quick code review.

### Steps
1. Add `ruff`/`flake8` and `black`.
2. Setup `pre-commit` and enforce checks.
3. Document git branching strategy.
4. Add `README` section for contributors.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`
