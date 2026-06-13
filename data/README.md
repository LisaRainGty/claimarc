# Data Directory

The raw, processed, final, cache, and experiment-output data for CLAIMARC are
kept outside git because they are large and may contain platform/private
research data.

Tracked code and docs refer to versioned artifact paths under this directory.
To reproduce an experiment, restore the corresponding data snapshot locally or
on the GPU server, then run the scripts documented in `docs/`.

Do not commit API keys, SSH credentials, raw product images, model checkpoints,
or generated training/evaluation artifacts to this repository.
