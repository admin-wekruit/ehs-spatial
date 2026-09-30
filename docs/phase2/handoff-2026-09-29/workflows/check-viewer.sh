#!/bin/zsh
# Human-check viewer: the layer server (8794, the mirror named in viewer-root.txt) runs inside this one preview,
# so closing a stray API tab can no longer kill it. Vite on 5185 proxies /fast to 8794 (the agents use 5173/8793).
S=/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad
W=/Users/adam/.codex/worktrees/panoptes-phase2-video-r4b-integrate
(cd $W && PYTHONPATH=. /Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python -m fast_report.layers serve "$(cat $S/viewer-root.txt)" --port 8794) &
LAYERS=$!
trap 'kill $LAYERS 2>/dev/null' EXIT INT TERM
cd $W/web && FAST_PORT=8794 npx vite --host 127.0.0.1 --port 5185 --strictPort
