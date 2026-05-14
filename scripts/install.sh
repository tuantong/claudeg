#!/bin/sh
# claudeg one-step installer for macOS and Linux.
set -eu

command -v curl >/dev/null 2>&1 || {
    echo 'curl is required but not installed' >&2
    exit 1
}
command -v tar >/dev/null 2>&1 || {
    echo 'tar is required but not installed' >&2
    exit 1
}

REPO="${CLAUDEG_REPO:-tuantong/claudeg}"   # set CLAUDEG_REPO=... to override
VERSION="${CLAUDEG_VERSION:-latest}"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS-$ARCH" in
    Darwin-arm64)  TARGET="aarch64-apple-darwin" ;;
    Darwin-x86_64) TARGET="x86_64-apple-darwin" ;;
    Linux-x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
    *) echo "unsupported OS/arch: $OS-$ARCH" >&2; exit 1 ;;
esac

if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/$REPO/releases/latest/download/claudeg-$TARGET.tar.gz"
else
    URL="https://github.com/$REPO/releases/download/$VERSION/claudeg-$TARGET.tar.gz"
fi

INSTALL_DIR="${CLAUDEG_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$INSTALL_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL"
curl -fSL "$URL" -o "$TMP/claudeg.tar.gz"
tar -xzf "$TMP/claudeg.tar.gz" -C "$TMP"
# Replace the binary atomically. On macOS, `cp` over an existing signed
# binary keeps the old code-signature cached and Gatekeeper SIGKILLs the
# new bits on launch — so we `rm` first to force a fresh inode + signature.
rm -f "$INSTALL_DIR/claudeg"
# Use cp+chmod rather than `install(1)` — busybox/Alpine ship a stripped
# `install` that does not accept the `-m` flag.
cp "$TMP/claudeg" "$INSTALL_DIR/claudeg"
chmod 0755 "$INSTALL_DIR/claudeg"

# On macOS, ad-hoc re-sign so the kernel trusts the new bytes. The release
# build is unsigned, so the existing signature (if any) is also ad-hoc.
if [ "$OS" = "Darwin" ] && command -v codesign >/dev/null 2>&1; then
    codesign --force --sign - "$INSTALL_DIR/claudeg" >/dev/null 2>&1 || true
fi

echo "Installed $INSTALL_DIR/claudeg"

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo "
NOTE: $INSTALL_DIR is not on your PATH.
Add this to your shell profile:
    export PATH=\"$INSTALL_DIR:\$PATH\"
" ;;
esac

echo "Running setup..."
exec "$INSTALL_DIR/claudeg" setup
