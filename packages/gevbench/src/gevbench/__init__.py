"""Benchmark harness for any `/v1/systemone` server: Jev, gev, or an LLM baseline.

Import submodules directly (`from gevbench import runner, metrics`); the package
binds nothing at import time, so importing one module does not pull in the rest.
"""
