#!/usr/bin/env bash
# Bump version for rift-local package
# Usage: ./scripts/bump-version.sh [major|minor|patch|dev]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Get current version
CURRENT_VERSION=$(grep -E '^version = ' pyproject.toml | cut -d'"' -f2)

echo "Current version: $CURRENT_VERSION"

# Parse version components
if [[ $CURRENT_VERSION =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(\.dev([0-9]+))?$ ]]; then
    MAJOR="${BASH_REMATCH[1]}"
    MINOR="${BASH_REMATCH[2]}"
    PATCH="${BASH_REMATCH[3]}"
    DEV="${BASH_REMATCH[5]}"
else
    echo "❌ Cannot parse version: $CURRENT_VERSION"
    exit 1
fi

# Determine new version based on argument
case "${1:-dev}" in
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    minor)
        NEW_VERSION="${MAJOR}.$((MINOR + 1)).0"
        ;;
    patch)
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
        ;;
    dev)
        if [ -n "$DEV" ]; then
            # Increment dev number
            NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}.dev$((DEV + 1))"
        else
            # Add dev0 to current version
            NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}.dev0"
        fi
        ;;
    release)
        # Remove dev suffix if present
        NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
        ;;
    *)
        # Custom version provided
        NEW_VERSION="$1"
        ;;
esac

echo "New version: $NEW_VERSION"
echo ""

# Confirm
read -p "Update version to $NEW_VERSION? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Update pyproject.toml
echo "Updating pyproject.toml..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Update __init__.py
echo "Updating src/rift_local/__init__.py..."
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" src/rift_local/__init__.py
rm src/rift_local/__init__.py.bak

# Verify consistency
VERSION_TOML=$(grep -E '^version = ' pyproject.toml | cut -d'"' -f2)
VERSION_INIT=$(grep -E '^__version__ = ' src/rift_local/__init__.py | cut -d'"' -f2)

if [ "$VERSION_TOML" != "$VERSION_INIT" ]; then
    echo "❌ Version update failed - mismatch detected!"
    echo "   pyproject.toml: $VERSION_TOML"
    echo "   __init__.py: $VERSION_INIT"
    exit 1
fi

echo ""
echo "✅ Version updated successfully: $CURRENT_VERSION → $NEW_VERSION"
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Review changes:"
echo "   git diff pyproject.toml src/rift_local/__init__.py"
echo ""
echo "2️⃣  Commit version bump:"
echo "   git add pyproject.toml src/rift_local/__init__.py"
echo "   git commit -m \"chore: bump version to $NEW_VERSION\""
echo ""
echo "3️⃣  Build and publish:"
echo "   ./scripts/build.sh"
echo ""
