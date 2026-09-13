# Security

## Reporting a vulnerability

Open the repository's **Security** tab and choose **Report a vulnerability**.
The report is private to you and the maintainers until an advisory is published.
Include the affected version, impact, and a minimal reproduction. Do not open
a public issue or include credentials in the report. Expect an acknowledgement
within a week.

## Scope

Reports may cover code execution through configs, checkpoints, or dataset
shards; path traversal; credential exposure; and bypassed integrity checks.
Model-quality issues belong in the issue tracker.

## Configs and data

Resolved configs and launch manifests are saved after `${VAR}` expansion.
The config validator rejects five key names: `token`, `secret`, `password`,
`api_key`, and `access_token`. It does not scan values or recognize other
credential fields. Keep credentials out of configs, including environment
substitutions; use workload identity or a credential file outside the repository.

Published indexes use relative POSIX shard keys. Tar members are read without
filesystem extraction. The object-store cache checks shard sizes against index
metadata.
