# GitHub Actions Workflows

## Workflows Overview

| Workflow | File | Description |
|----------|------|-------------|
| **CI** | `ci.yml` | Continuous Integration - tests, lint, build |
| **CD Release** | `cd-release.yml` | Continuous Deployment - publish releases |
| **Security** | `security.yml` | Security scanning - dependencies, code, docker |
| **Documentation** | `docs.yml` | Auto-generate and deploy docs |

## CI Pipeline (`ci.yml`)

Triggered on every push and pull request.

### Jobs:
1. **Lint** - Code quality checks (flake8, black, isort, mypy)
2. **Unit Tests** - Run on Python 3.10, 3.11, 3.12
3. **Integration Tests** - Test component interactions
4. **API Tests** - End-to-end API testing
5. **Coverage** - Generate and upload coverage report
6. **Docker Build** - Build and test Docker image
7. **Docker Publish** - Push to Docker Hub (main branch only)
8. **Deploy** - Deployment placeholder

## CD Pipeline (`cd-release.yml`)

Triggered on release publication or manual dispatch.

### Jobs:
1. **Release** - Version bump and tagging
2. **Docker Multi-Arch** - Build for amd64 and arm64
3. **Publish PyPI** - Upload to Python Package Index
4. **Security** - Final security scan
5. **Notify** - Send release notifications

## Security Pipeline (`security.yml`)

Triggered weekly and on PRs.

### Jobs:
1. **Dependency Scan** - Check for vulnerable packages (safety)
2. **Docker Scan** - Scan Docker image (Trivy)
3. **CodeQL** - Static code analysis

## Documentation Pipeline (`docs.yml`)

Triggered on documentation changes.

### Jobs:
1. **Generate Docs** - Create HTML documentation
2. **Deploy** - Publish to GitHub Pages
3. **Check Docs** - Validate documentation coverage

## Required Secrets

Configure these in GitHub Repository Settings → Secrets:

| Secret | Description | Required For |
|--------|-------------|--------------|
| `DOCKERHUB_USERNAME` | Docker Hub username | Docker publish |
| `DOCKERHUB_TOKEN` | Docker Hub access token | Docker publish |
| `CODECOV_TOKEN` | Codecov upload token | Coverage report |
| `PYPI_API_TOKEN` | PyPI API token | PyPI publish |

## Optional Secrets

| Secret | Description | Used By |
|--------|-------------|---------|
| `HEROKU_API_KEY` | Heroku API key | Deploy to Heroku |
| `HEROKU_APP_NAME` | Heroku app name | Deploy to Heroku |
| `AWS_ACCESS_KEY_ID` | AWS credentials | Deploy to ECS |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | Deploy to ECS |
| `SLACK_WEBHOOK` | Slack webhook URL | Release notifications |

## Manual Triggers

Some workflows support manual triggering:

```bash
# Release specific version
gh workflow run cd-release.yml -f version=1.0.0

# Run security scan
gh workflow run security.yml

# Generate documentation
gh workflow run docs.yml
```

## Skipping Workflows

Add to commit message to skip CI:
```
git commit -m "fix: typo [skip ci]"
```

## Badge URLs

Add these badges to your README:

```markdown
[![CI](https://github.com/your-username/document-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/document-processor/actions/workflows/ci.yml)
[![Security Scan](https://github.com/your-username/document-processor/actions/workflows/security.yml/badge.svg)](https://github.com/your-username/document-processor/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/your-username/document-processor/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/document-processor)
```
