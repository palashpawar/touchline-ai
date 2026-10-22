#!/usr/bin/env bash
# Fetch the Creative Commons demo clips from Wikimedia Commons (see data/CLIPS.md for attribution).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/clips
UA="TouchlineAI-research/1.0"
[ -f data/clips/france_germany_2018_phase1_ccbysa.ogv ] || curl -fsSL -A "$UA" -o data/clips/france_germany_2018_phase1_ccbysa.ogv \
  "https://upload.wikimedia.org/wikipedia/commons/b/ba/Match_de_football_France-Allemagne_-_16_octobre_2018_-_Phase_de_jeu_%281%29.ogv"
[ -f data/clips/latvia_gibraltar_2026-03-31_cc0.webm ] || curl -fsSL -A "$UA" -o data/clips/latvia_gibraltar_2026-03-31_cc0.webm \
  "https://upload.wikimedia.org/wikipedia/commons/0/06/Latvia-Gibraltar_football_2026-03-31.webm"
shasum -a 256 data/clips/*
