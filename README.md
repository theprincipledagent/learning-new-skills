# Skill RL
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A RL/APO system that automatically evolves [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill files by iteratively solving, evaluating, and learning from in-use rollouts.

This is part of the Principled Agent blog.

**[Read the full story here](https://www.principledagent.com/).**

## 📝 Blog Post Series & Code

Each post in the series has a corresponding **GitHub Release** that contains a permanent snapshot of the code exactly as it was at the time of publishing.

| Post | Title | Blog Post Link | Code Release |
| :--- | :--- | :---: | :---: |
| #1 | Why AI-as-a-Judge is Hard | [Read Post](https://theprincipledagent.com/2026/04/07/why-ai-as-a-judge-is-hard-learning-new-skills-1/) | [v1.1]

---

## How It Works

Skill RL runs an evolutionary loop where Claude Code solves research questions, gets evaluated on its performance, and then uses its failures to generate improved skill files for the next round. Skills are markdown files with YAML frontmatter that get injected into Claude Code's system prompt to guide its behavior.

Each epoch follows three phases:

1. **Actor** -- Claude Code runs inside Docker containers, solving sampled GAIA questions using the current set of skills. Answers are extracted and checked against ground truth.
2. **Evaluator** -- Each rollout is scored across five dimensions (helpfulness, accuracy, reasoning quality, tool selection, knowledge application). Can run in blind mode or with ground-truth diagnostics.
3. **Evolver** -- The worst-performing rollouts (bottom percentile) are fed to Claude, which generates new or modified skill files. A trust region constraint prevents overly drastic changes. If overall accuracy drops, skills are rolled back to the last checkpoint.

```
            ┌──────────┐
            │  Sample   │
            │ Questions │
            └────┬─────┘
                 │
                 ▼
          ┌─────────────┐
          │    Actor     │  Claude Code in Docker
          │  (parallel)  │  solves questions
          └──────┬──────┘
                 │ rollouts
                 ▼
          ┌─────────────┐
          │  Evaluator   │  Score each rollout
          └──────┬──────┘
                 │ scores + diagnostics
                 ▼
          ┌─────────────┐
          │   Evolver    │  Evolve skills from
          │              │  worst rollouts
          └──────┬──────┘
                 │ updated skills
                 ▼
          ┌─────────────┐
          │  Rollback?   │  Revert if accuracy
          │              │  degraded
          └──────┬──────┘
                 │
                 └──── next epoch ───►
```

## Requirements

- Python 3.11+
- Docker (daemon must be running)
- An Anthropic API key or Claude Code OAuth token

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Training

```bash
# Basic run
skill-rl --api-key $ANTHROPIC_API_KEY

# With options
skill-rl \
  --api-key $ANTHROPIC_API_KEY \
  --model sonnet \
  --num-epochs 10 \
  --questions-per-epoch 100 \
  --max-parallel-actors 5 \
  --use-benchmark-score

# Resume from a specific epoch
skill-rl --api-key $ANTHROPIC_API_KEY --start-epoch 5
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | | Anthropic API key |
| `--oauth-token` | | Claude Code OAuth token (alternative to API key) |
| `--model` | `haiku` | Model to use (`haiku`, `sonnet`, `opus`) |
| `--num-epochs` | `10` | Number of training epochs |
| `--questions-per-epoch` | `100` | Questions sampled per epoch |
| `--max-actor-turns` | `25` | Max agentic turns per question |
| `--max-parallel-actors` | `5` | Concurrent Docker containers |
| `--trust-region-threshold` | `0.3` | Max allowed skill change ratio |
| `--bottom-percentile` | `0.2` | Fraction of worst rollouts used for evolution |
| `--rollback-threshold` | `0.05` | Accuracy drop that triggers a rollback |
| `--actor-timeout` | `600` | Timeout per question (seconds) |
| `--use-benchmark-score` | off | Use ground truth for failure diagnostics |
| `--start-epoch` | `0` | Epoch to resume from |

### Generalization Testing

Test evolved skills on a held-out set of questions (no overlap with training):

```bash
python -m skill_rl.generalization_test \
  --api-key $ANTHROPIC_API_KEY \
  --num-questions 50 \
  --test-seed 123
```

### Analysis Scripts

```bash
# Compare a question's rollouts across epochs
python compare_rollouts.py TASK_ID
python compare_rollouts.py --list

# Correctness heatmap across questions and epochs
python plot_correctness.py --show

# Evaluation score vs accuracy correlation plots
python plot_eval_vs_accuracy.py --show
```

## Skill Format

Skills are stored in `skills/<name>/SKILL.md` as markdown with YAML frontmatter:

```markdown
---
name: web-research-strategy
description: Systematic approach to web research tasks
---

1. **Start broad, then narrow**: Begin with general searches...
2. **Cross-reference sources**: Always verify claims...

<!-- [EVOLUTION cycle 3] Added verification step after finding too many hallucinated sources -->
```

The evolver appends HTML comments to track what changed and why across evolution cycles.

## Output Structure

Each epoch writes to `data/epochs/epoch_N/`:

```
epoch_N/
├── actor_system_prompt.txt     # Rendered prompt with skills
├── evaluations.json            # Scores and diagnostics
├── evolution.json              # Skill diffs and reasoning
├── metrics.json                # Epoch statistics
└── rollouts/
    └── <task_id>/
        ├── <task_id>.json      # Full rollout data
        ├── prompt.txt          # The question
        ├── run.sh              # Docker command used
        └── [attached files]    # Any files needed for the question
```

Cross-epoch history is saved to `data/history.json`.

## Architecture

```
src/skill_rl/
├── main.py          # Entry point, epoch loop, rollback logic
├── actor.py         # Runs Claude Code in Docker, parses outputs
├── evaluator.py     # Scores rollouts (blind or with ground truth)
├── evolver.py       # Generates evolved skills from failures
├── skills.py        # Skill loading/saving, trust region enforcement
├── gaia.py          # GAIA dataset loading, stratified sampling, answer checking
├── docker_utils.py  # Docker image building and container management
├── config.py        # Configuration dataclass and CLI arg mapping
└── metrics.py       # Accuracy tracking and progress reporting
```

The actor phase runs in Docker for isolation. The evaluator and evolver run on the host via subprocess calls to the Claude CLI.
