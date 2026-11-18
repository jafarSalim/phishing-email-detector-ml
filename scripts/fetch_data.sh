#!/usr/bin/env bash
# Download the raw messages used by the experiments into data/raw/ (about 600 MB).
# phishing_pot is CC BY-NC 4.0 (non-commercial research); SpamAssassin is CC0 via stdlib-js.
# A fixed random subset is used so results are reproducible: see experiments/subsets.txt.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw
if [ ! -d phishing_pot ]; then
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/rf-peixoto/phishing_pot
  (cd phishing_pot && git checkout 78be141545ed9e56eefb857808ac78e83eb81b81 -- $(grep '^email/' ../../../experiments/subsets.txt))
fi
if [ ! -d spamassassin ]; then
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/stdlib-js/datasets-spam-assassin sa
  (cd sa && git checkout 5b9f4463de0fc01649e3379dde724d41057c8286 -- $(grep '^data/' ../../../experiments/subsets.txt))
  mkdir -p spamassassin && cp -r sa/data/* spamassassin/ && rm -rf sa
fi
echo "done: $(find . -type f | wc -l) files"
