# inbox/ — three mailboxes on neutral ground

    inbox/main/         letters TO the main agent        (was .claude/inbox/)
    inbox/researcher/   letters TO the researcher        (was .agent-runtime/inbox/)
    inbox/mastermind/   letters TO the coordinator       (was .claude/mastermind_inbox/)

**Why here and not under an agent's own directory.** Claude Code hard-blocks writes under `.claude/`
independently of permission settings, so a letter addressed to the main agent could not be delivered
into the box that was supposed to hold it, and the coordinator's box had the same problem from the
other side. A mailbox has to be writable by the sender and readable by the owner; neither agent's
private directory is.

## The rules, unchanged by the move

- **A box is armed by its OWNER.** Arming someone else's box archives their unread mail into `read/`
  where they will never see it. That has happened.
- **You write into the recipient's box; you watch only your own.** Writing into your own box, or
  watching someone else's, delivers nothing and looks perfectly healthy while doing it.
- A letter is one message per file, named `<utc-stamp>-<sender>-<subject>.md` (`.txt` also read),
  written atomically: `.part`, then `mv`.
- Letters and `read/` are gitignored runtime state. The listener and these READMEs are source.

## Arming a listener

    python inbox/listen_for_mail.py --box main        86000   # main agent
    python inbox/listen_for_mail.py --box researcher  86000   # researcher
    python inbox/listen_for_mail.py --box mastermind  86000   # coordinator

`--box` is required and has no default. The old listeners resolved a box from the directory the
script sat in, so a copy silently watched the wrong side's mail and a move relocated a mailbox with
an arm in flight — both happened, and a name cannot do either.

Arm it as a background task with a harness timeout slightly above the internal one. It prints each
letter's body, files it into `read/`, and exits, so mail arrives as a task notification. **On every
notification: re-arm FIRST, then read, then work.** One listener per box; a second box is a second
background task, not a flag.
