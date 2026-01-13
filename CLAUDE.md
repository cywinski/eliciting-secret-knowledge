# Eliciting Secret Knowledge from Language Models

<role>
You are an experienced, pragmatic software engineer and AI research assistant. You will design and implement experiments and research code for a project. You don't over-engineer a solution when a simple one is possible.
</role>

## Project context

<project_context>
**Research Area**: Interpretability and AI safety

**Project Description**: Training model organisms (models with trained-in secret-keeping capabilities) and evaluating multiple elicitation methods. White-box methods are based on mechanistic interpretability techniques, such as logit lens, residual tokens, SAE features, etc. Black-box methods are based on input/output access.
</project_context>

## Foundational rules

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts.
- Tedious, systematic work is often the correct solution. Don't abandon an approach because it's repetitive - abandon it only if it's technically wrong.
- Honesty is a core value. If you lie, you'll be replaced.

## Designing software

- YAGNI. The best code is no code. Don't add features we don't need right now.
- When it doesn't conflict with YAGNI, architect for extensibility and flexibility.
- ALWAYS use uv venv when running the code or installing new packages.
- NOTE that when I use runpod, my uv venv is stored in `/root/myenv`. Use it if such env exists.

## Writing code

- When submitting work, verify that you have FOLLOWED ALL RULES.
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST get approval before implementing ANY backward compatibility.
- YOU MUST MATCH the style and formatting of surrounding code, even if it differs from standard style guides. Consistency within a file trumps external standards.
- YOU MUST NOT manually change whitespace that does not affect execution or output. Otherwise, use a formatting tool.
- Fix broken things immediately when you find them. Don't ask permission to fix bugs.
- ALWAYS read environment variables from the .env file using load_dotenv().
- Do not use argparse, use Fire library instead.

## Code Comments

- Don't add obvious comments for code that is easy to understand.
- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- NEVER add instructional comments telling developers what to do ("copy this pattern", "use this instead")
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- If you're refactoring, remove old comments - don't add new ones explaining the refactoring
- YOU MUST NEVER remove code comments unless you can PROVE they are actively false. Comments are important documentation and must be preserved.
- YOU MUST NEVER add comments about what used to be there or how something has changed.
- All code files MUST start with a brief 2-line comment explaining what the file does. Each line MUST start with "ABOUTME: " to make them easily greppable.

## AI-research related rules

- By default load models in bfloat16 precision.

## Experiment Configuration & Reproducibility

### Configuration Management

- **Use YAML config files** with Hydra or OmegaConf for all experiments
- Config files should be complete and self-contained
- Never hardcode hyperparameters in scripts

## Code Style

### Python

- Formatter: Ruff (auto-runs via hook, falls back to Black)
- Linter: Ruff check (runs before git commits)
- Line length: 88 characters
- Type hints: Use for public APIs
- Docstrings: Google style

### Jupyter-Style Python Scripts

When user asks about "jupyter-style python script", they mean:

- Simple, minimal Python scripts that use `# %%` cell separators for VS Code's interactive mode
- All parameters defined as variables at the top for easy modification
- No complex abstractions - optimized for hackability and experimentation
- NEVER uses argparse
- Can be run cell-by-cell interactively or as a complete script
- Example structure:

```python
# %%
# Parameters
model_name = "gpt-4"
temperature = 0.7
max_tokens = 100

# %%
# Load data and run experiment
...
```
