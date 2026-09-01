#!/usr/bin/env bash
# .agent-runtime/kimi-session-start.sh — SessionStart hook (wired in ~/.kimi-code/config.toml).
#
# Kimi auto-loads the project's AGENTS.md; what it does NOT see are the upward
# CLAUDE.md tiers (global ~/.claude, parent dirs) that Claude Code loads. Print
# those, plus a mailbox digest and listener liveness, so the hook's stdout lands
# in context as the session's opening banner. Keep it compact — this is every
# session's first context.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
INBOX="$SCRIPT_DIR/inbox"

# Regenerate the stack + CLAUDE.md sync (report only the drift line, if any).
bash "$SCRIPT_DIR/agent_md_sync.sh" | grep -v "^  -" || true

# --- Upward tiers (the part Kimi does not auto-load) ---
for f in "${HOME}/.claude/CLAUDE.md" "$REPO/../CLAUDE.md"; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    echo "=== CLAUDE.md tier: ${f/#$HOME/~} ==="
    cat "$f"
    echo ""
  fi
done

# --- Mailbox digest ---
if [ -d "$INBOX" ]; then
  mapfile -t LETTERS < <(ls "$INBOX"/*.txt "$INBOX"/*.md 2>/dev/null)
  if [ "${#LETTERS[@]}" -gt 0 ]; then
    echo "=== INBOX: ${#LETTERS[@]} unread ==="
    printf '  %s\n' "$(basename -a "${LETTERS[@]}")"
  else
    echo "=== INBOX: empty ==="
  fi
  HB="$INBOX/.listener_heartbeat"
  if [ -f "$HB" ] && [ $(( $(date +%s) - $(date -r "$HB" +%s) )) -lt 30 ]; then
    echo "listener: ARMED"
  else
    echo "listener: NOT ARMED — arm with: python .agent-runtime/listen_for_mail.py (background task)"
  fi
fi
