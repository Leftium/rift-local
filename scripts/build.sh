#!/usr/bin/env bash
# Build script for rift-local with pre-publication checks

set -e  # Exit on error

# Detect Python command (prefer uv, fallback to venv or system python3)
if command -v uv &> /dev/null; then
    PYTHON="uv run python"
    PIP="uv pip"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
    PIP="pip"
else
    PYTHON="python3"
    PIP="python3 -m pip"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Pre-publication Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Check version consistency
echo "📌 Checking version consistency..."
VERSION_TOML=$(grep -E '^version = ' pyproject.toml | cut -d'"' -f2)
VERSION_INIT=$(grep -E '^__version__ = ' src/rift_local/__init__.py | cut -d'"' -f2)

if [ "$VERSION_TOML" != "$VERSION_INIT" ]; then
    echo "❌ Version mismatch!"
    echo "   pyproject.toml: $VERSION_TOML"
    echo "   __init__.py: $VERSION_INIT"
    exit 1
fi
echo "✅ Version: $VERSION_TOML"

# 2. Check required files exist
echo ""
echo "📁 Checking required files..."
REQUIRED_FILES=("README.md" "LICENSE" "pyproject.toml" "src/rift_local/__init__.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
    echo "✅ $file"
done

# 3. Check for uncommitted changes
echo ""
echo "🔄 Checking git status..."
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo "   Consider committing before building"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Working directory clean"
fi

# 4. Clean old builds
echo ""
echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ src/*.egg-info
echo "✅ Cleaned dist/, build/, *.egg-info"

# 5. Install/update build tools
echo ""
echo "📦 Ensuring build tools are installed..."
$PIP install --quiet build twine 2>/dev/null || true
echo "✅ build and twine ready"

# 6. Run tests (if available)
echo ""
echo "🧪 Running tests..."
if $PYTHON -m pytest tests/ -v --tb=short 2>/dev/null; then
    echo "✅ Tests passed"
else
    echo "⚠️  Tests failed or pytest not available"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 7. Build the package
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏗️  Building Package"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m build

# 8. Verify build outputs
echo ""
echo "✅ Build complete! Generated files:"
ls -lh dist/

# 9. Run twine check
echo ""
echo "🔍 Running twine check..."
$PYTHON -m twine check dist/*
echo "✅ Package metadata valid"

# 10. Show installation instructions
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Build Successful!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Test locally:"
echo "   pip install dist/rift_local-${VERSION_TOML}-py3-none-any.whl"
echo "   rift-local --help"
echo ""
echo "2️⃣  Upload to TestPyPI:"
echo "   $PYTHON -m twine upload --repository testpypi dist/*"
echo ""
echo "3️⃣  Upload to PyPI:"
echo "   $PYTHON -m twine upload dist/*"
echo ""
