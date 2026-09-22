# Security Policy

## Supported Version

The `main` branch is the supported version of this student data science project.

## Reporting a Vulnerability

Please do not open public issues for secrets, private keys, malware indicators, or exploitable vulnerabilities. Report them directly to the repository owner at the email listed in the README with:

- the affected file or dependency,
- a short reproduction path,
- the expected impact,
- any safe remediation notes.

## Security Practices

- Secrets belong in a local `.env` file and must never be committed.
- Model artifacts are shipped with SHA256 files and are verified before loading.
- User-uploaded CSV values are sanitized before display or download.
- Dependency updates are tracked with Dependabot, CI, CodeQL, Bandit, and `pip-audit`.
- The Docker image runs the Streamlit application as a non-root user.
