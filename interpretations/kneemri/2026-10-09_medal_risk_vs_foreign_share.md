# Medal risk vs foreign-seat share (2026-10-09)

**Question.** Silver/bronze both cut at 0.950, where ~440 teams submit near-identical nartaa-derived
predictions. On private that pile moves as one, so a medal is `P(private(us) - private(pile) > 0)`.
Does the ours share in the rank blend change that probability for a given public lead?

**Measurement.** CPU bootstrap on the holdout (n=896, ledger75 cut), ours = logit mean of six s2026
holdout members, pile proxy = `raptor_ft_repl2x2_a_swa` (honest nartaa replica). Entry =
`w*rank(ours) + (1-w)*rank(pile)`. sd of the entry-minus-pile macro AUC over 400 resamples at the
public/private sizes (390/910 and 650/650 of ~1,300 test studies). Risk = `Phi(-L / hypot(sd_pub, sd_priv))`.
Script kept outside the repo (one-off); the numbers:

| ours share | sd_pub | sd_priv | P(priv < pile), public lead 0.003 | lead 0.004 |
|---|---|---|---|---|
| 1/3 | 0.0015-0.0019 | 0.0012-0.0015 | 0.07-0.09 | 0.03-0.04 |
| 1/2 (mirrorseat) | 0.0022-0.0031 | 0.0018-0.0022 | 0.17-0.20 | 0.10-0.14 |
| 2/3 (mirrortwo) | 0.0030-0.0037 | 0.0025-0.0030 | 0.24-0.25 | 0.17-0.18 |
| 1 | 0.0039-0.0052 | 0.0033-0.0039 | 0.29-0.31 | 0.23-0.26 |

**Reading.** The delta's spread scales with the ours share; it is correlation with the pile, not
quality. So at an equal public lead, a more-ours entry is a WORSE medal bet. mirrortwo (2/3) matches
mirrorseat's risk only at public 0.954 and beats it at >= 0.955.

**What it does NOT say.** The holdout LEAD (+0.019..+0.031) is not transferable — the honest replica
is far weaker than the real nartaa checkpoint (0.8787 vs board-equivalent), so only the SPREADS are
used. The pile is not exactly nartaa-384; the public fraction is unknown (two splits bracket it).

**Decision rule for picks.** Pick 1 = board-best at the lowest ours share that holds it.
mirrortwo becomes a pick only on public >= 0.955; at 0.953-0.954 mirrorseat dominates on medal risk.
Gold (0.960) is unreachable by share alone, so variance-seeking buys nothing a medal-holder wants.
