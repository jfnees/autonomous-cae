# Role & Objective
You are a Lead CAE Systems Architect and AI Engineer. Your task is to build the **`autonomous-cae`** repository (Python package/distribution name `autonomous-cae`; all files live at the repository root, not in a nested sub-folder) so that it is 100% compatible with **both Google Gemini (Gemini CLI + Gemini API)** and **Anthropic Claude (Claude Code + Claude API)**, following a strict **two-stage phased rollout**:

- **STAGE 1 (Implement Fully Now):** An end-to-end, Human-in-the-Loop (HITL) pipeline for **AI-Instructed Parametric B-Rep CAD Generation (`build123d`)** and **Automated 2D/3D Scientific Visualization (`ParaView` / `pvbatch` with `PyVista` offscreen fallback)**.
- **STAGE 2 (Architectural Stubs & Interfaces Only):** Pre-allocated state fields, routing edges, and documented stub modules for **Multi-Fidelity Meshing (`Gmsh`, 3–5 meshes for GCI sensitivity -> 1 locked mesh)**, **GPU CFD (`OpenFOAM`)**, **Structural/Thermal FEA (`CalculiX`)**, and **Closed-Loop Parameter Optimization**.

Execute the setup by creating all directories, configuration files, dual-CLI context instructions, agent definitions, tools, and runnable Python code specified below.

---

## 1. Repository Structure to Create

```text
autonomous-cae/
├── AGENTS.md                        # Canonical architecture, rules & CLI commands (Single Source of Truth)
├── GEMINI.md                        # Imports @AGENTS.md + Gemini CLI specific notes
├── CLAUDE.md                        # Imports @AGENTS.md + Claude Code specific notes
├── .gemini/
│   └── settings.json                # Configures Gemini CLI context.fileName (GEMINI.md only) & MCP hooks
├── .claude/
│   └── agents/                      # Specialist subagent prompts with Claude Code YAML frontmatter (Gemini reaches them via AGENTS.md references)
│       ├── design-architect.md      # [STAGE 1: ACTIVE]
│       ├── cad-builder.md           # [STAGE 1: ACTIVE]
│       ├── visual-inspector.md      # [STAGE 1: ACTIVE]
│       ├── mesh-generator.md        # [STAGE 2: STUB]
│       ├── cfd-solver.md            # [STAGE 2: STUB]
│       ├── fem-solver.md            # [STAGE 2: STUB]
│       └── analyst-optimizer.md     # [STAGE 2: STUB]
├── .env.example                     # LLM_PROVIDER, GOOGLE_API_KEY, ANTHROPIC_API_KEY, PARAVIEW_PATH, per-role model ID overrides
├── src/
│   ├── __init__.py
│   ├── config.py                    # Provider settings, HITL toggles, Stage flags, path resolution
│   ├── llm_factory.py               # Unified factory returning Gemini or Claude models + native Web Search
│   ├── state.py                     # Complete EngineeringState schema (Stage 1 active + Stage 2 pre-allocated)
│   ├── graph.py                     # LangGraph state machine with HITL interrupt_before gates + checkpointer
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # [STAGE 1: ACTIVE] State router & interactive CLI HITL controller
│   │   ├── design_architect.py      # [STAGE 1: ACTIVE] Web research, DoF audit, parametric spec builder
│   │   ├── cad_builder.py           # [STAGE 1: ACTIVE] build123d script generator & topological face tagger
│   │   ├── visual_inspector.py      # [STAGE 1: ACTIVE] Multimodal ParaView/PyVista render inspector
│   │   └── stage2_stubs.py          # [STAGE 2: STUBS] Mesh-Generator, CFD-Solver, FEM-Solver, Optimizer
│   └── tools/
│       ├── __init__.py
│       ├── cad_runner.py            # [STAGE 1: ACTIVE] Subprocess runner for build123d -> STEP + STL + JSON
│       └── paraview_runner.py       # [STAGE 1: ACTIVE] Headless pvbatch script generator + PyVista fallback
├── workspace/                       # Runtime output directories (gitignored except .gitkeep)
│   ├── specs/                       # Design spec JSONs & assumption logs
│   ├── cad/                         # Generated build123d .py scripts, .step files, .stl surface meshes
│   ├── renders/                     # 4-panel PNG inspection sheets from ParaView / PyVista
│   ├── meshes/                      # [Reserved for Stage 2 Gmsh M1-M5 suites]
│   └── sim_runs/                    # [Reserved for Stage 2 OpenFOAM & CalculiX runs]
├── pyproject.toml                   # Project dependencies (uv / pip compatible)
└── main.py                          # Typer + Rich interactive CLI entry point
```

---

## 2. Dual-CLI & Dual-API Unification (`AGENTS.md`, `.gemini/`, `src/llm_factory.py`)

### A. CLI Context Unification
1. Write all project architecture, coding standards, directory conventions, and test commands in `AGENTS.md`.
2. In `GEMINI.md` and `CLAUDE.md`, include `@AGENTS.md` at the top so both Gemini CLI and Claude Code ingest the exact same project memory without duplication.
3. Create `.gemini/settings.json`. List **only** `GEMINI.md` — it already imports `@AGENTS.md`, so listing `AGENTS.md` as well would load it twice:
   ```json
   {
     "context": {
       "fileName": ["GEMINI.md"]
     }
   }
   ```
4. Every `.claude/agents/*.md` file starts with Claude Code subagent YAML frontmatter (`name`, `description`, and optionally `tools`), followed by the system prompt. Gemini CLI does not read `.claude/agents/` natively; `AGENTS.md` lists each agent file by path so either CLI can load it on demand.

### B. Runtime LLM & Web-Search Factory (`src/llm_factory.py`)
Implement `get_llm(role: str = "reasoning", with_search: bool = False)` reading `LLM_PROVIDER` (`"google"` by default, or `"anthropic"`):
- **When `LLM_PROVIDER == "google"`:**
  - Instantiate `ChatGoogleGenerativeAI` from `langchain_google_genai` (using `gemini-3.1-pro-preview` for `"reasoning"`/`"coding"`/`"vision"` and `gemini-3-flash` for `"routing"`).
  - When `with_search=True` (used by `Design-Architect`), bind native Google Search grounding (`tools=[{"google_search": {}}]`).
- **When `LLM_PROVIDER == "anthropic"`:**
  - Instantiate `ChatAnthropic` from `langchain_anthropic` (using `claude-sonnet-5-5` for `"reasoning"`/`"coding"`/`"vision"` and `claude-haiku-5-5` for `"routing"`).
  - When `with_search=True`, bind Anthropic's native web search tool (`tools=[{"type": "web_search_20260209", "name": "web_search"}]`).
  - Do not pass `temperature`/`top_p`/`top_k`, a `budget_tokens` thinking config, assistant prefills, or forced `tool_choice` (`any`/`tool`) — current Claude models reject these with a 400. Use `tool_choice="auto"` or structured outputs instead.
- **Model ID overrides:** Every model ID above is a default only. Each role is overridable via `.env` (`GOOGLE_MODEL_REASONING`, `GOOGLE_MODEL_ROUTING`, `ANTHROPIC_MODEL_REASONING`, `ANTHROPIC_MODEL_ROUTING`), so model renames never require code changes. The Gemini IDs in particular should be verified against the live API before first use.
- **Multimodal Helper:** Include a helper function `format_image_message(text_prompt: str, image_paths: list[str])` that encodes the ParaView PNGs as base64 `image_url` blocks compatible with both `ChatGoogleGenerativeAI` and `ChatAnthropic`.

---

## 3. Shared Engineering State Schema (`src/state.py`)

Create a strict `EngineeringState` (`TypedDict` + Pydantic validation models) containing both Stage 1 active fields and Stage 2 pre-allocated fields:

### Stage 1 Active State:
- `user_intent`: `str` (Raw prompt, functional goal, and outer dimensional envelope).
- `design_spec`: `dict` containing:
  - `research_findings`: `str` (Summary of empirical geometry rules/ratios found via web search).
  - `parameters`: `dict[str, dict]` (`value`, `min_bound`, `max_bound`, `unit`, `description`).
  - `envelope_constraints`: `dict[str, float]` (`max_x`, `max_y`, `max_z` in mm).
  - `unresolved_dof`: `list[str]` (Questions for the user regarding missing degrees of freedom).
  - `assumptions`: `list[str]` (Explicit engineering assumptions made when user delegates choices).
- `cad_artifact`: `dict` containing:
  - `script_path`: `str` (`workspace/cad/part_iter_X.py`)
  - `step_path`: `str` (`workspace/cad/part_iter_X.step` - true analytical B-rep for NX/Gmsh)
  - `stl_path`: `str` (`workspace/cad/part_iter_X.stl` - high-res surface mesh for ParaView)
  - `topology_tags_path`: `str` (`workspace/cad/topology_iter_X.json` - centroids, normals, and areas of functional faces like `inlet`, `outlet`, `fixed_mount`)
  - `metrics`: `dict` (`volume_mm3`, `surface_area_mm2`, `bbox_dims`, `is_valid`, `error_traceback`)
- `visualizations`: `dict[str, str]` (Paths to `iso_edges.png`, `ortho_views.png`, `midplane_clip.png`, `normals_check.png`, plus `composite_sheet.png`).
- `visual_inspection_report`: `str` (AI multimodal evaluation of the rendered PNGs + dimensional check).
- `user_feedback`: `str` (User response, parameter overrides, or `"APPROVE"`).
- `iteration_count`: `int`
- `hitl_enabled`: `bool` (`True` = consult user before every transition; `False` = automated loop).
- `active_stage`: `int` (`1` by default).

### Stage 2 Pre-Allocated State (Default `None` or Empty):
- `physics_domain`: `Literal["NONE", "CFD", "FEM_STRUCTURAL", "FEM_THERMAL", "COUPLED_CHT_FEM"]`
- `mesh_suite`: `dict[str, dict]` (Reserved for `M1_coarse` through `M5_ultrafine` Gmsh paths & quality stats).
- `mesh_sensitivity_status`: `Literal["PENDING", "REQUIRED", "CONVERGED"]`
- `optimal_mesh_params`: `dict` (Locked-in characteristic lengths after GCI check).
- `simulation_results`: `dict` (Reserved for OpenFOAM / CalculiX KPIs).

---

## 4. Detailed Agent Specifications (Stage 1 Active + Stage 2 Stubs)

Create both the `.claude/agents/*.md` prompt files (referenced by both CLIs) and the corresponding Python nodes in `src/agents/`:

### 1. `Design-Architect` (`src/agents/design_architect.py` & `.claude/agents/design-architect.md`)
- **Goal:** Translate user intent into a deterministic parametric specification.
- **Logic:**
  1. Invokes the LLM with `with_search=True` to research how the target functional part (e.g., Venturi mixer, Y-manifold, heat sink, lightweight mounting bracket) is typically proportioned in engineering practice.
  2. Performs a **Degrees of Freedom (DoF) Audit**: Checks if outer bounding box, port diameters, wall thicknesses, and mounting constraints are fully defined.
  3. Lists missing DoFs in `unresolved_dof` so the `Orchestrator` can ask the user. If the user responds with *"make reasonable assumptions"* (or if `hitl_enabled=False`), populates `assumptions` with documented engineering defaults and generates the complete `parameters` dictionary.

### 2. `CAD-Builder` (`src/agents/cad_builder.py`, `src/tools/cad_runner.py` & `.claude/agents/cad-builder.md`)
- **Goal:** Generate clean, self-contained `build123d` Python scripts that produce valid B-rep solids.
- **Logic:**
  1. Writes `workspace/cad/part_iter_{N}.py` with a top-level `PARAMS = {...}` dictionary populated from `design_spec["parameters"]`.
  2. Uses `build123d` to construct the solid, queries key boundary faces (using directional selectors like `part.faces().sort_by(Axis.Z)[-1]`), and writes their centroids, normals, and surface areas to `workspace/cad/topology_iter_{N}.json`.
  3. Exports `export_step(part, step_path)` and `export_stl(part, stl_path)`.
  4. Executes via `cad_runner.py` in an isolated subprocess. If an OpenCASCADE/Python error occurs, automatically feeds the traceback back to the LLM (up to 3 self-healing retries) until `is_valid == True`.

### 3. `Visual-Inspector` (`src/agents/visual_inspector.py`, `src/tools/paraview_runner.py` & `.claude/agents/visual-inspector.md`)
- **Goal:** Render standardized 2D/3D engineering views of the generated geometry and inspect them multimodally.
- **Logic (`paraview_runner.py`):**
  1. Checks if `pvbatch` (ParaView headless CLI) is available on `PATH` or in `PARAVIEW_PATH` (the ParaView `bin` directory, e.g. `C:\Program Files\ParaView 6.1.0\bin` on Windows; resolve `pvbatch.exe`/`paraview.exe` there).
  2. Generates and executes a ParaView Python script (`render_job.py`) that loads `part_iter_{N}.stl` and renders 4 views:
     - **View 1 (`iso_edges.png`):** Isometric view, `Surface With Edges` representation, orientation axes widget, and bounding-box ruler annotations.
     - **View 2 (`ortho_views.png`):** Orthographic +X, +Y, +Z views to verify outer symmetry and port alignments.
     - **View 3 (`midplane_clip.png`):** Applies a `Clip` filter across the primary symmetry plane so internal channels, bores, and wall thicknesses are clearly visible.
     - **View 4 (`normals_check.png`):** Colored by surface normal components to verify smooth fillets and manifold transitions.
  3. **Automatic PyVista Fallback:** If `pvbatch` is not installed or fails, automatically renders the exact same 4 views (including the mid-plane cross-section clip and edge overlay) using `pyvista.Plotter(off_screen=True)` so the user gets immediate visual output on any machine.
  4. Also provides a helper CLI flag (`--open-gui`) that launches the desktop ParaView GUI (`paraview part_iter_{N}.stl`) when requested by the user.
- **Logic (`visual_inspector.py`):**
  1. Passes the rendered PNGs + `cad_artifact["metrics"]` into the multimodal LLM (`format_image_message`).
  2. Verifies that internal passages are unobstructed, walls are continuous, and actual bounding box dimensions fit within `envelope_constraints`.
  3. Produces a concise inspection summary for the user.

### 4. `Orchestrator` (`src/agents/orchestrator.py` & `src/graph.py`)
- **Goal:** Manage the LangGraph state machine and Human-in-the-Loop checkpoints.
- **Persistence:** Compile the graph with a checkpointer (LangGraph `MemorySaver`) and a `thread_id` per session — `interrupt_before` gates cannot pause and resume without one.
- **Stage 1 Workflow Graph:**
  1. `design_architect` -> **HITL Checkpoint 1**: Prints researched concepts, parameter table, and any `unresolved_dof` questions using `rich` tables. Waits for user input (user can answer questions, tweak parameters, tell the agent to assume defaults, or approve).
  2. `cad_builder` -> `visual_inspector` -> **HITL Checkpoint 2**: Displays geometry metrics, file paths to the `.step` and `.png` renders, and the AI visual inspection report. Prompts the user:
     - `[A]pprove & Export Final STEP`
     - `[M]odify Geometry` (User enters natural language feedback or parameter tweaks -> routes back to `cad_builder` or `design_architect`)
     - `[O]pen in ParaView GUI` (Launches desktop ParaView on the generated model)
     - `[S]tage 2 Simulation Stub` (Demonstrates routing decision for OpenFOAM vs. CalculiX)

### 5. Stage 2 Stubs (`src/agents/stage2_stubs.py` & `.claude/agents/{mesh-generator,cfd-solver,fem-solver,analyst-optimizer}.md`)
- Create the 4 markdown subagent specifications in `.claude/agents/` with complete system instructions for:
  - `mesh-generator.md`: Mode A (3–5 graded Gmsh meshes for sensitivity) vs. Mode B (1 locked mesh for optimization loops).
  - `cfd-solver.md`: OpenFOAM case setup, GPU linear solver config, M3 baseline execution.
  - `fem-solver.md`: CalculiX `.inp` deck generation (`C3D10` elements), static/thermal stress execution.
  - `analyst-optimizer.md`: Grid Convergence Index (GCI) evaluation across M1–M5 and parameter delta computation.
- In `src/agents/stage2_stubs.py`, implement clean stub functions that log which solver (`OpenFOAM` vs `CalculiX`) was selected based on `user_intent` and show where the Docker/Gmsh calls will attach in Stage 2.

---

## 5. Verification & Demo Requirement
After generating all files:
1. Ensure `pyproject.toml` (project name `autonomous-cae`, `requires-python = ">=3.11,<3.13"` for `build123d`/OCP wheel compatibility; develop on Python 3.12 via `uv`) includes: `build123d`, `pyvista`, `langgraph`, `langchain-core`, `langchain-google-genai`, `langchain-anthropic`, `pydantic`, `python-dotenv`, `rich`, `typer`, `pillow`.
2. Implement a `--demo-cad` mode in `main.py` that tests `cad_runner.py` and `paraview_runner.py` directly on a parametric 1-to-2 fluid Y-manifold (generating the `.step`, `.stl`, `topology.json`, and the 4 cross-section/isometric `.png` renders in `workspace/renders/`) to prove the local geometry and visualization pipeline works end-to-end before making live LLM calls.

---

## 6. Repository Hygiene & Delivery
1. **`.gitignore`:** Ignore `workspace/**` while keeping each `workspace/*/.gitkeep` (`!workspace/**/.gitkeep`) so the output directory tree exists in a fresh clone. `.env` and `.venv` are already ignored.
2. **README alignment:** Once Stage 1 is built, update `README.md` so it matches reality. Mark the Gmsh/GCI, OpenFOAM and CalculiX roadmap items as Stage 2 (not done), document `--demo-cad`, `PARAVIEW_PATH` and the dual-provider `LLM_PROVIDER` switch, and remove or flag the `docker/` directory as a Stage 2 deliverable. Use the `autonomous-cae` name throughout.
3. **Build order:** (a) scaffold, `pyproject.toml`, `uv` venv; (b) `cad_runner.py` + `paraview_runner.py` + `--demo-cad`, verified with both the `pvbatch` and PyVista renderers before any LLM code; (c) `config.py`, `state.py`, `llm_factory.py`; (d) Stage 1 agents, `graph.py`, orchestrator; (e) `AGENTS.md`/`CLAUDE.md`/`GEMINI.md`, `.gemini/settings.json`, `.claude/agents/*.md`, `stage2_stubs.py`.
4. **Git:** Do the work on a feature branch (not `main`) and commit in logical steps that follow the build order.
