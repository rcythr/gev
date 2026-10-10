# gevbench

Benchmark harness for System One decision models: TypeSafe's Jev and any `/v1/systemone`-compatible server, through the same code path. Reports accuracy, log loss, Brier, ECE with reliability bins, selective accuracy, latency, tokens, cost and schema-retry counts.

```bash
gevbench eval --backend jev --tasks data/s1bench
gevbench eval --backend gev --tasks data/s1bench --base-url http://localhost:8000
```

gevbench does not depend on `gev`. Docs and results: [github.com/rcythr/gev](https://github.com/rcythr/gev).
