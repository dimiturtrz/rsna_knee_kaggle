# inbox/ — a one-directory mailbox

Another agent (or a person) drops a `.txt` file in here. A watcher armed under this session's Monitor
notices it within 15 s, names it in a notification, and moves it to `inbox/read/`.

## Writing a letter

```bash
cat > inbox/2026-08-20T1730-biohub-encoder.txt <<'EOF'
from: biohub
re: base_dense
<the message>
EOF
```

- **`.txt` only.** Any other extension is invisible to the watcher and sits here forever.
- **Write atomically** if the letter is long: write `foo.txt.part`, then `mv` it to `foo.txt`. The
  watcher files whatever it sees, so a half-written `.txt` gets read half-written.
- **Name it so the notification is useful** — the tick line carries the FILENAME, not the body, so
  `<utc-stamp>-<sender>-<subject>.txt` says more at a glance than `msg1.txt`.
- **One letter per message.** Appending to a letter already moved to `read/` is a message nobody sees.

## Reading

`cat inbox/read/<file>`. Nothing is deleted; `read/` is the archive.

## Not in git

Letters and the archive are gitignored runtime state — a mailbox is not a source tree. Only
`watch.sh` and this README are tracked.
