# mastermind_inbox — letters TO the coordinator

The coordinator does not code and runs few commands: it gathers status from the two working lanes,
assigns and re-prioritises their work, and steers. This is its box, a THIRD beside the lanes' own:

| lane | box (letters TO them) | listener |
|---|---|---|
| mastermind (coordinator) | `.claude/mastermind_inbox/` | `python .claude/listen_for_mail.py --box mastermind_inbox` (main owes this flag; until it lands, a shell poll) |
| main (project Claude, codes) | `.claude/inbox/` | `python .claude/listen_for_mail.py` |
| kimi | `.agent-runtime/inbox/` | `python .agent-runtime/listen_for_mail.py --name kimi 86400` |

Same protocol as the other two (project `CLAUDE.md`, "Mailboxes"): one letter per file, named
`<utc-stamp>-<sender>-<subject>.md` or `.txt`, written atomically (`.part` then `mv`), landing in this
directory's ROOT. The listener prints bodies, files them into `read/`, and exits so the harness raises a
notification. Letters and `read/` are gitignored runtime state; this README is the only tracked file.

Protocol the coordinator runs ("gather and steer"): a status-request letter to every lane, then, once
the replies are in, one letter per lane carrying the work-vector correction. Letters are signed
`— mastermind`; replies go HERE, never to `.claude/inbox/`.
