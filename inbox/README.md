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

## The switch — `inbox/agents.csv` and `python inbox/agents.py`

A lane can be turned OFF and back ON, and the two halves of "is it alive" are kept in two files
because config and runtime truth must not share a home:

- **`inbox/agents.csv`** — human-owned config, git-tracked, one row per box, `active` 1 or 0. The
  toggle CLI is its only programmatic writer.
- **`inbox/<box>/.status`** — runtime, gitignored, written by that box's OWN listener alone: pid,
  when it armed, when it last delivered, how much. One writer per file, so nothing needs a lock.

The listener reads the registry on arm AND on every poll. An inactive box prints
`box <x> is INACTIVE in inbox/agents.csv (<note>) — not arming` and exits 2, so switching a lane off
also stops one that is already running. **Unread letters are never archived while inactive** — they
stay in the box and are delivered when the lane is armed again. Off means silenced, never lost.

    python inbox/agents.py --list                          # config JOINED with runtime evidence
    python inbox/agents.py --box researcher --off "reason"  # silence a lane, with the reason recorded
    python inbox/agents.py --box researcher --on
    python inbox/agents.py --check researcher               # exit 0 active, 1 not — gate a letter on it

`--list` flags the two lies a flag alone cannot catch: `active=1` with no `.status`, or one older
than a day, reads **STALE — flagged active, no live listener** (the dead-watcher failure wearing a
green flag); `active=0` with letters in the box reads **OFF, N letters waiting**.

**What the switch does NOT do.** Switching a box off silences its MAIL. It does not stop that side's
wakeup loop — that is the owner's `/loop` control — and it does not let one side arm or disarm
another's listener. A box is still armed by its owner, one listener per box.
