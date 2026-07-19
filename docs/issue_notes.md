# Issue Notes

Short record of problems encountered while bringing up the project. No secrets
or official task contents are included.

- GitHub clone attempted to use a dead `127.0.0.1` proxy. Cleared proxy settings
  and used SSH authentication for the remote server.
- Local Windows WSL optional-feature installation was unreliable. Moved the
  CUDA validation and training workflow to the AutoDL Linux instance.
- AutoDL package downloads timed out through stale proxy/mirror settings.
  Cleared proxy variables, selected a reachable index, and increased timeout.
- RTX PRO 6000 Blackwell runtime required the proven combination of Python
  3.12, PyTorch 2.11 CUDA 13.0, and vLLM 0.24.0. verl is installed with
  `--no-deps` so its old metadata does not replace the working vLLM runtime.
- FlashAttention installation hit NVCC/wheel compatibility problems. The
  runtime was validated independently before enabling the optimized path.
- A tau2 smoke task received reward zero after the agent invented an optional
  database field. Tool arguments must match the user request exactly.
- Training must use verl's managed vLLM rollout engine. The standalone port-8000
  server is reserved for independent inference checks.
- verl worker count did not cap total episode concurrency. Added a per-worker
  semaphore so the global target is eight concurrent episodes, plus three
  full-episode attempts with exponential backoff.
- tau2 may return an empty observation when the simulator/orchestrator fails.
  Added typed infrastructure checks and fresh-episode retries.
- tau2 converts the special `done` tool into `agent_stop` and returns no normal
  ToolMessage. It is now recorded as a control action and excluded from
  environment-tool process reward.
- `TauInfrastructureError` could not cross the Ray process boundary. Added an
  explicit pickle reconstruction protocol and a regression test.
- tau2 NL assertions defaulted to GPT-4.1 and failed without an OpenAI key.
  User simulation uses `deepseek-v4-pro`; the NL judge is explicitly
  configured as `deepseek-v4-pro` for all experiment groups.
- LiteLLM may warn that DeepSeek model pricing is not mapped. This affects cost
  reporting only and does not invalidate rollout or reward computation.
- The first E0 run set rollout `n` but verl validation read `val_kwargs.n`, so
  it produced one trial per official task. The launcher now sets both fields;
  the archived run remains a valid one-trial diagnostic.
- The earlier multi-domain E0 diagnostic mixed airline, retail, and telecom
  and therefore is not a result under the final protocol. Formal E0-E5 runs
  now use only the official tau2 airline train/test split. Synthetic, retail,
  and telecom utilities are retained for auxiliary analysis only.
- Resuming a checkpoint with `SWANLAB_RUN_ID` but without `SWANLAB_RESUME`
  made SwanLab reject the run before the first resumed step. The verl launcher
  now selects `SWANLAB_RESUME=must` whenever an explicit run ID is supplied
  and rejects contradictory resume settings before Ray starts.
