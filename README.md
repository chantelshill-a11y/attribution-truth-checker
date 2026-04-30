# Attribution Truth-Checker

A validation layer for marketing attribution models. Runs simulated geo-lift experiments on synthetic data with known ground truth, calculates measured incrementality per channel, compares it against what an attribution model claims, and produces an executive narrative explaining the gap.

The premise: attribution models output credit. They do not output truth. This system tests whether the credit matches reality.

## Status

Phase 1 in progress. Synthetic data generator under construction. Geo-lift engine, comparison layer, narrative, and self-evaluation harness are next.

## Quick start

Coming soon. Will be a single CLI command that runs the generator, the analysis, and produces an HTML report.
