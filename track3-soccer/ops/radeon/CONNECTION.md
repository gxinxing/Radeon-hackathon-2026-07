# Radeon connection record

The latest recorded runtime used the Radeon cloud JupyterLab instance and the
remote project path `/workspace/amd-physical-ai-soccer/`.

This path was also used for the 2026-08-04 AMD GPU validation. The older
`/workspace/radeon-repo` path found in archived scripts and handoffs is not a
current synchronization target.

The endpoint and fingerprint must still be checked before each new synchronization
session. Obtain them from the provider console, verify them through a trusted
channel, and keep the private key under the user's SSH directory rather than this
repository.

Do not bypass host verification. After verification, record only the approved
public fingerprint and endpoint here; never record a private key or access token.

The removed legacy backup script is deliberately not archived as executable
source: it used `StrictHostKeyChecking=no`, swallowed transfer errors, and could
place a GitHub token inside a remote URL.
