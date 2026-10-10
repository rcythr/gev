#!/usr/bin/env bash
# Package and check the e4b-instruct release on Modal, pull it and upload it to a
# private Hugging Face repo.
#
#   REPO=<org>/gev scripts/publish_hf.sh
#   REPO=<org>/gev DRY_RUN=1 scripts/publish_hf.sh   # everything but the upload
set -euo pipefail

PRESET="${PRESET:-e4b-instruct}"
NAME="${NAME:-gev}"
REPO="${REPO:?set REPO to the Hub repo to publish to, e.g. REPO=<org>/gev}"

cd "$(dirname "$0")/.."
release="weights/$NAME"

# A token in .env (HF_TOKEN=...) wins over the one `hf auth login` stored.
if [[ -f .env ]] && grep -q '^HF_TOKEN=' .env; then
  set -a
  # shellcheck source=/dev/null
  source <(grep '^HF_TOKEN=' .env)
  set +a
fi

hf auth whoami >/dev/null || { echo "not logged in to the Hub: run 'hf auth login'" >&2; exit 1; }

modal run modal/app.py::export_checkpoint --preset "$PRESET" --name "$NAME"
# Load the release as a user will before anything is pulled or published.
modal run modal/app.py::check_release --name "$NAME"
# The target must be an existing directory: given a missing path, `volume get`
# writes every file of the release onto that one path.
mkdir -p weights
modal volume get --force gev-checkpoints "releases/$NAME" weights/

echo "release ready in $release:"
ls -la "$release"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN set: not uploading to $REPO"
  exit 0
fi

uv run gev release publish "$release" --repo "$REPO" --private
