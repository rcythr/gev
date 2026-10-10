# gev

An open System One decision model. Give it a state and typed questions (yes/no, choice, score), and it answers all of them in one forward pass with calibrated probabilities over exactly the options you supplied. A LoRA adapter on Gemma 4 E4B, served over TypeSafe's `/v1/systemone` protocol. A fork of [lev](https://github.com/InterfazeAI/lev).

```bash
pip install "gev[serve] @ git+https://github.com/rcythr/gev#subdirectory=packages/gev"
```

```python
import gev

model = gev.load(
    "checkpoints/gev-e4b-instruct", model_id="google/gemma-4-E4B-it", prompt_style="gemma"
)
```

The core install is pure Python (schema, prompt layouts, router, calibration); `[train]` adds the model stack, `[serve]` the HTTP server. Benchmarks, training and design: [github.com/rcythr/gev](https://github.com/rcythr/gev).
