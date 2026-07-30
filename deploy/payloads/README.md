# Launch payloads

Known-good `POST /load` bodies, one per model that is fiddly enough to be worth
pinning. Not used by any script — these are for a human (or an agent) restoring
a host by hand after a deploy, a reboot, or an agent restart.

```sh
scp deploy/payloads/dspark.json sparky:/tmp/
ssh sparky 'curl -s -X POST localhost:4400/load \
  -H "Content-Type: application/json" --data-binary @/tmp/dspark.json'
```

Each file is the **effective** profile — the agent's defaults already applied,
exactly as `GET /slots` reports it. So it is explicit and reproducible rather
than relying on whatever the adapter defaults happen to be at the time. They
round-trip: what you `POST` is what `/slots` gives back.

They deliberately carry **no sampling** (`temperature`, `top_p`, penalties).
Airo sets sampling per request, and per-request values override server defaults
anyway. Baking `temperature: 1.1` + `repetition_penalty: 1.05` into the DSpark
load produced gibberish — repetition penalty interacts badly with its
speculative decoding — so `--generation-config vllm` keeps the server neutral.
Note the vLLM adapter maps no sampling keys at all (see
`Engine.Vllm.honored_profile_keys/0`); a genuine server-side default has to ride
in `extra_argv` as `--override-generation-config`.

No secrets belong in here. Unlike `deploy/hosts/*.env` (gitignored, may carry
`AIRO_AGENT_TOKEN`), these files are tracked — keep them to model ids and launch
knobs.

## Files

| File | Host | Notes |
| --- | --- | --- |
| `dspark.json` | `sparky` (+ `sparky2` as rank 1) | Two-host TP load, `nnodes: 2`. Reloading is a **cluster** operation: the head's wrapper starts rank 1 on the worker over SSH, so posting this to sparky alone brings up both ranks. Any agent restart on either Spark drops it and it must be reposted. |
