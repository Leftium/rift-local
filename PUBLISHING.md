# Publishing rift-local to PyPI

## Quick Start

```bash
# 1. Build and test
./scripts/build.sh

# 2. Upload to TestPyPI (optional but recommended)
uv run python -m twine upload --repository testpypi dist/*

# 3. Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rift-local

# 4. Upload to PyPI
uv run python -m twine upload dist/*
```

## First-Time Setup

### 1. Create PyPI Accounts

- **TestPyPI**: https://test.pypi.org/account/register/
- **PyPI**: https://pypi.org/account/register/

### 2. Generate API Tokens

For both TestPyPI and PyPI:
1. Go to Account Settings → API tokens
2. Click "Add API token"
3. Name: `rift-local` (or "All projects" scope)
4. Scope: Choose project or entire account
5. Copy the token (starts with `pypi-`)

### 3. Configure Authentication

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

Set permissions:
```bash
chmod 600 ~/.pypirc
```

## Build Script Features

The `scripts/build.sh` script automatically:

✅ Checks version consistency (`pyproject.toml` ↔ `__init__.py`)  
✅ Verifies required files exist  
✅ Warns about uncommitted changes  
✅ Cleans old builds  
✅ Runs tests  
✅ Builds both wheel and source distribution  
✅ Validates package metadata with `twine check`  
✅ Shows next steps

## Version Management

Current version: `0.1.0.dev0` (development release)

### Update version in TWO places:

1. `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.0"
   ```

2. `src/rift_local/__init__.py`:
   ```python
   __version__ = "0.1.0"
   ```

### Version Strategy

- `0.1.0.dev0` — Development release (TestPyPI testing)
- `0.1.0` — First stable release
- `0.1.1` — Bug fixes (patch)
- `0.2.0` — New features (minor)
- `1.0.0` — Stable API (major)

## Testing the Package

### Local Installation Test

```bash
# Create clean venv
python3 -m venv /tmp/test_rift_local
source /tmp/test_rift_local/bin/activate

# Install from local build
pip install dist/rift_local-*.whl

# Test CLI
rift-local --help
rift-local list

# Test Python API
python3 -c "
import rift_local
print(f'Version: {rift_local.__version__}')
from rift_local.models.registry import list_models
print(f'Models: {len(list_models())}')
"
```

### TestPyPI Installation Test

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            rift-local

# Note: --extra-index-url needed for dependencies not on TestPyPI
```

## Publishing Checklist

Before uploading to PyPI:

- [ ] All tests pass
- [ ] Version updated in both places
- [ ] README.md up to date
- [ ] CHANGELOG updated (if exists)
- [ ] Git committed and tagged
- [ ] Tested on TestPyPI
- [ ] Verified installation in clean environment

## Post-Publication

### Tag the release

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### Create GitHub Release

1. Go to https://github.com/Leftium/rift-local/releases
2. Click "Draft a new release"
3. Choose tag: `v0.1.0`
4. Release title: `v0.1.0`
5. Add release notes
6. Publish release

## Troubleshooting

### `externally-managed-environment` error

The build script automatically handles this by using `uv` if available. 

If needed, use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/build.sh
```

### Upload fails with 403

- Check API token is correct
- Verify package name not already taken
- Ensure token has correct scope

### Package name conflict

If `rift-local` is taken, consider:
- `rift-local-server`
- `riftlocal`
- `rift-inference`

Change in `pyproject.toml`:
```toml
[project]
name = "rift-local-server"
```

## Useful Commands

```bash
# View package contents
tar -tzf dist/rift_local-*.tar.gz | head -20
unzip -l dist/rift_local-*.whl

# Check package on PyPI
open https://pypi.org/project/rift-local/

# Check download stats (after 24-48 hours)
open https://pepy.tech/project/rift-local
```
