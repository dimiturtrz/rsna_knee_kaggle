# What the host has actually ruled on external data — forum sweep, 2026-09-13

Swept the competition discussion forum for every thread touching external data, licences and the
winner obligation. Kaggle discussion pages are JS-rendered and `WebFetch` returns only the title;
everything below was read through `https://r.jina.ai/<url>`, which renders them.

The point of the sweep: before asking the host anything, find out whether it has already been asked.
It has — three times, by three different people, and the question we care about is still open.

## The four threads that carry a HOST answer

### 733965 — the host's own clarification post. Po-Hao "Howard" Chen, competition host.

This is the only place the host has written at length, and it settles the NON-COMMERCIAL axis:

> readily accessible datasets with non-commercial use restrictions are generally acceptable under
> competition rules, since receiving prize money does not constitute commercial exploitation

with the caveat that a team must still comply with each dataset's own licence terms. Same post,
on sending competition data to third-party LLMs:

> submitting Competition Data, including report text, to an external LLM or API for inference or
> other computational processing (for example, extracting labels from reports) will not, by itself,
> be considered prohibited PRIVATE SHARING

and, asked directly, "You can use LLM API, such as those from OpenAI, to read the reports to
generate the labels."

**Consequence for us.** The NC half of the Rijeka question is answered and it is answered our way.
CC BY-NC is not a bar. It also retroactively confirms the repo's 2026-09-07 retraction of the line
"NC and ND clauses almost certainly violate rule 2.6a" — the host does not read 2.6a that way either.
And it confirms the CC0 LLM-soft-label route (`raptor/labels/labels_llm_soft.csv`) is clean at the
source, not merely clean by licence.

### 734109 — KneeCoT. Asked by Mazzutti (2nd on the board). Kaggle staff (javaduke95) answered in 19 days.

> we decided to not allow the use of KneeCoT dataset due to the required data agreement

The refusal is on the AGREEMENT. Not on the CC BY-NC licence — which 733965 says is fine anyway.
The axis the host polices is the GATE.

### 733652 — "Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels". Fernando Faria.

Names **MRNet, fastMRI+, the Osteoarthritis Initiative and SKM-TEA explicitly** and asks whether a
research-use agreement breaks "equally accessible at no cost". The host REPLIED in this thread — and
redirected to 733965, which answers the LLM half and the NC half and says nothing about gating.

So the four corpora have been put to the host by name, and the host's answer addressed a different
question than the one asked.

Community replies worth knowing: NNMax cites §6 that external data is acceptable unless specifically
prohibited; Oleksii Zhukov reads §4.b as barring route (a) for LLMs (superseded by the host's own
733965); Mazzutti draws the distinction this repo already runs on — permission to TRAIN is not
eligibility to WIN.

## The three threads with NO host answer — this is the gap

### 737950 — "External data behind an institutional data use agreement — does the KneeCoT ruling generalize?" Ercan Gurvit, posted 16 days ago.

**Our exact question, asked by someone else, sixteen days ago, with no host reply.** The thread ran to
nine community posts and the host never entered. What the participants concluded on their own:

- Tucker Arrants: NC is fine, **ND is the problem** — "Most external datasets that are relevant here
  are ND, and training a model on images, then releasing those weights as required by Kaggle winning
  solutions, would be a derivative and hence prohibited." That is this repo's own 2.5 reading,
  arrived at independently by a competitor.
- Fernando Faria asks for the distinction that would actually decide it: **click-through agreement vs
  emailed/signed agreement**, across MRNet, fastMRI, OAI and SKM-TEA. Unanswered.
- Doublej37 and junicus push a different case: public pretrained CHECKPOINTS whose training data is
  private or unclear (MRI-CORE named). Also unanswered.
- PC Jimmmy's observation, which is correct and which this repo already runs on: the restriction only
  binds WINNERS.

### 738156 — "are ungated, openly-licensed OAI-derivative datasets allowed?" 14 days ago.

Asks the sharpest version: does "publicly and freely available" attach to the ARTEFACT you use or to
the PROVENANCE behind it, citing OAIZIB-CM on HuggingFace (ungated, CC BY-NC). One comment, deleted.
No host answer. This is the thread whose answer would settle Rijeka, whose Zenodo record and Kaggle
mirror are both ungated.

### The Rule 2.6(a) thread (Demir Poyraz Elcin) and "Question about Winner License and Apache 2.0 Pretrained Models" (owaowata, 6 days ago)

Both live, both on the same axis, neither with a host ruling visible.

## Reading

The host has answered the two EASY questions (NC, LLM APIs) at length and in participants' favour,
and answered the one HARD question only in the single instance where a dataset was named with its
agreement attached (KneeCoT, refused). The general rule — does a data use agreement disqualify a
corpus, and does a click-through differ from a signature — has been asked by three separate people
over sixteen days and never answered.

Two things follow.

1. **A fourth "does it generalize?" thread joins a queue of three that were ignored.** Posting one is
   not the move. What has DRAWN a host answer, twice, is a SINGLE NAMED DATASET with its access
   mechanism stated (KneeCoT), and a question posted where the host was already writing (733965).

2. **The silence is itself informative, and it cuts our way for the ungated sets.** 2.6b says external
   data is "acceptable unless specifically prohibited by the Host". One dataset has been specifically
   prohibited. Rijeka and KMAR carry no agreement of any kind, so the KneeCoT reasoning does not reach
   them, and the NC clause the host has now explicitly blessed is Rijeka's only other exposure beyond ND.

## What this changes about our position

- **KMAR-50K**: CC BY 4.0, no gate, no NC, no ND. Clean on every axis the host has ruled on. Unchanged,
  and it remains the only merge shippable through 2.5.
- **Rijeka**: ungated (Zenodo + public Kaggle mirror) and its NC clause is now explicitly fine per
  733965. Sole remaining exposure is **ND against rule 2.5**, which is exactly what Tucker Arrants
  independently identified as the binding constraint. That is a question about WEIGHTS-AS-DERIVATIVE,
  not about access, and it is the one worth putting to the host.
- **MRNet / fastMRI / SKM-TEA**: still point AGAINST under the KneeCoT reasoning, and the host has
  declined three invitations to say otherwise. Treat them as instruments already cashed
  (`slots.py:341-344`: second site +0.0043, third −0.0024), never as seats.
