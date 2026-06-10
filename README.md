# TOPOLOGIX

TDA-based drug-protein interaction screening. Bipartite simplicial complex at the
drug-protein interface, opposition-distance metric, persistent homology yielding
Internuclear Persistence Contours (IPCs). MVP target: hERG cardiotoxicity screening.

## Architecture

| Module | Role |
|---|---|
| `complex` | Bipartite simplicial complex construction (drug + protein interface) |
| `metric` | Opposition-distance metric |
| `homology` | Filtration + persistent homology -> IPCs (Ripser/Gudhi) |
| `features` | IPC feature extraction for downstream classification |
| `screen` | End-to-end hERG screen: structure in, risk score out |
| `llm` | LLM router (Groq -> Mistral -> OpenRouter) + daily budget guard |
| `wolfram` | Wolfram derivation harness: symbolic derivation + validation co-processor |
| `data` | Loaders (PDB, ligand, hERG benchmark) |

## Execution model

Issues are tracked in Linear (Princia / TOPOLOGIX). Two issue classes:

- **Derivation** issues produce the mathematical core. Acceptance = passing a
  constraint set + the validation gate, not implementing a known spec. These use
  the Wolfram harness.
- **Engineering** issues implement known specs. Acceptance = tests pass.

## The validation gate

`benchmarks/harness.py`. IPC-derived features must beat an RDKit-descriptor +
XGBoost baseline on a held-out hERG split. If they do not, the core formulation
is wrong and the chain halts at validation. This gate is what makes the IP real.

## Dev

```
pip install -e ".[dev]"
pytest -q
```
