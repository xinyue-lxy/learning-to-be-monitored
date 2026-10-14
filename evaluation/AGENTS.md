# Release maintenance

- Read README.md and docs/METRICS.md before changing evaluation behavior.
- Preserve pinned task/monitor/verifier profiles, prompt/data hashes, output parsing and support gates. Treat new behavior as an explicitly versioned policy.
- Keep native/common analysis in one instance-first estimator; include valid zeros and preserve NA.
- Keep runtime outputs, weights, credentials, private paths and experiment-specific registrations outside the source tree.
- Never edit source checkpoint tensors or sealed task/Y inputs during replay.
- CPU planning/analysis must not import GPU libraries or initialize CUDA.
- Run synthetic tests after edits; CPU success does not replace a GPU canary.
- Do not install dependencies, download weights, submit jobs or publish code/models without authorization.
- Use non-preemptible typed H200/B200 allocations for GPU validation, never login nodes or RTX resources. Submission belongs to the user.
- Review third-party code, dataset and model licenses separately before publication.
