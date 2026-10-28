# Reproducibility and artifact scope

This portfolio release intentionally separates the method artifact from third-party environments and large generated assets.

Included:

- Core model, estimator, interaction-design, and trainer code
- Focused component tests
- Final aggregate tables and seed-level summaries
- Paper and supplement sources/PDFs

Excluded:

- Third-party benchmark source trees and simulator installations
- Raw environment datasets and generated trajectories
- Model checkpoints, replay buffers, tensor dumps, and run logs
- Intermediate or superseded experiment directories

To reproduce full campaigns, install each benchmark from its upstream project, implement its native observation/action adapter, and use the frozen protocols described in the domain reports and paper. The CSV files in this repository are archived outputs, not regenerated fixtures.

No API keys, credentials, machine-specific virtual environments, or private datasets are included.
