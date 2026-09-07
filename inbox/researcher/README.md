# inbox/researcher/ — letters TO the researcher

Was `.agent-runtime/inbox/`. Anyone may write here; only the researcher watches it.

    python inbox/listen_for_mail.py --box researcher 86000

Letters and `read/` are gitignored. See `../README.md` for the rules that apply to all three boxes.

This lane can be switched off (`python inbox/agents.py --box researcher --off "reason"`) and back on. While
it is off the listener refuses to arm and letters STAY here undelivered — off is silenced, never lost.
`python inbox/agents.py --list` says whether the lane is on AND whether a listener is really alive.
