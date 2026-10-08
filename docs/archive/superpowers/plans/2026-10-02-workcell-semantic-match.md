# Workcell spatial-semantic experiment

Goal: test whether HOV-SG-inspired visual feature fusion and geometric association connect our existing 52 workcell objects to natural-language concepts and evidence for future EHS policy applicability. Preserve the models, coordinate system, and caliper.

1. Freeze a category vocabulary, fixed retrieval queries, geometry thresholds and proxy catalog references before inference. References are evaluation-only; never feed entity names or expected classes into encoders or association.
2. Reuse `route_jev.tight_crop` and `route_jev.Encoder`. One ephemeral Modal job, exactly two A100-80GB GPUs, runs PE-Core-L and SigLIP2 in parallel. No generative VLM. Cached depth and masks are inputs; this measures incremental semantic latency, not fresh photo-to-model runtime.
3. Compare single-view crops, multiview HOV-inspired fusion, and label-blind geometry-gated semantic association. Require at most one observation per photo in a geometric group. Measure proxy class agreement and cross-view identity agreement/coverage, including hard negatives.
4. Save source hashes, scores, support observations, thresholds, timing and spend ledger. Cosines are not probabilities; catalog references are not ground truth. Existing geometry and model-scale status remain unchanged.
5. Add optional experiment UI to the existing report: fixed natural-language queries select actual objects; show crops, competing labels, conditional geometry and missing policy evidence. No invented policy source, threshold, applicability or verdict.
6. Run self-check, integration validation, independent adversarial review and browser selection/caliper checks; publish only `workcell-photo-direct/` and source branch. Document failures as well as successful matches.

Sources: https://hovsg.github.io/ ; https://arxiv.org/html/2403.17846v1 ; HOV-SG `sam_clip_feats_extractor.py`. Our encoder and per-object aggregation differ from HOV-SG; this is an engineering adaptation, not a reproduction benchmark.
