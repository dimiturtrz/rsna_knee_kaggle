# fastMRI and MRNet — competition-use licence audit

## Question

Can the project use fastMRI and MRNet as external training data for the RSNA Knee Abnormalities Detection Kaggle competition without violating Kaggle rule 2.6a/2.6b?

## Kaggle rule reminder

From `comp/COMPETITION_RULES.txt` Section 2.6:

> a. You may use data other than the Competition Data ("External Data") to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria ...

> b. The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions ... their use must be "reasonably accessible to all" and of "minimal cost".

In practice this means: public, equally accessible, no-cost, and no licence clause that bars competition use.

---

## MRNet

### Licence text

The Stanford MRNet Dataset Research Use Agreement is reproduced in the website source ([GitHub commit `a07223d`](https://github.com/stanfordmlgroup/stanfordmlgroup.github.io/commit/a07223dfda18ebae2a09a8c43e11e7365c3ccc90.patch)) and currently appears on the download page. Key clauses:

> 1. Permission is granted to view and use the MRNet Dataset without charge for **personal, non-commercial research purposes only**. Any commercial use, sale, or other monetization is prohibited.

> 4. **YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY** of any portion or all of the MRNet Dataset to others without specific prior written permission from the School of Medicine.

> 6. You must not modify, reverse engineer, decompile, or **create derivative works** from the MRNet Dataset.

> 10. Any violation of this Research Use Agreement ... shall be grounds for immediate termination of use ...

### Competition-use assessment

| Kaggle 2.6a test | MRNet status |
|---|---|
| Publicly available | Yes, after registration |
| Equally accessible | Yes, but registration required |
| At no cost | Yes |
| No restrictive licence | **No** — non-commercial + no redistribution + no derivative works |

The non-commercial clause alone is commonly read as incompatible with Kaggle 2.6a. The no-derivative-works clause is even more restrictive for a machine-learning competition, because training a model on the data and using its weights in a submission is arguably a derivative work. The no-redistribution clause also prevents packaging MRNet into a Kaggle dataset for the kernel environment.

### Verdict

**High risk.** MRNet should not be shipped in a Kaggle submission without explicit host approval or a written licence clarification from Stanford. This aligns with the existing bead `knee-ombx` and the note in `research/deep_dives/2026-09-02_external_knee_mri_data_access_audit.md`.

---

## fastMRI

### Licence text

The fastMRI paper (Knoll et al., 2020, [PMC6996599](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/)) states:

> The dataset is open and available to anyone for educational and research purposes, with no requirement to submit a research proposal to access the data. However, users must sign up and agree to the data sharing agreement. The complete data sharing agreement is available on the download webpage. **In general, while we do not allow any commercial use of the dataset itself, we do not explicitly discourage the development or testing of software, algorithms, or other intellectual property with the dataset.**

The full data sharing agreement is only available after registration at `https://fastmri.med.nyu.edu/`.

### Competition-use assessment

| Kaggle 2.6a test | fastMRI status |
|---|---|
| Publicly available | Yes, after registration |
| Equally accessible | Yes |
| At no cost | Yes |
| No restrictive licence | **Partial** — non-commercial restriction on the dataset itself, but explicit allowance for algorithm/software development |

fastMRI is less restrictive than MRNet: it explicitly permits "development or testing of software, algorithms, or other intellectual property with the dataset." A Kaggle submission is an algorithm, not a redistribution of the dataset. The remaining concern is the non-commercial clause. Kaggle competitions are non-commercial for participants (no purchase of data, no direct sale of predictions), but the host/platform is commercial. Most competition participants treat fastMRI-style NC clauses as acceptable for submission because the dataset itself is not being sold or redistributed.

### fastMRI+

The pathology annotations (`fastMRI+`) are hosted in the Microsoft `fastmri-plus` GitHub repo under a Microsoft CLA for contributions; the underlying fastMRI licence still governs the images. The annotations themselves appear to be a community/Microsoft contribution and are likely usable under the same terms as the base dataset, but the exact licence for the annotation CSVs should be checked in the repo.

### Verdict

**Lower risk than MRNet, but not zero risk.** The explicit allowance for algorithm development weighs in favor of competition use. The conservative move is to disclose fastMRI as external data in the submission and to avoid redistributing the raw images/annotations inside a public Kaggle dataset. If the host objects, the objection will almost certainly be to the NC clause; there is no way to remove that except by asking NYU/fastMRI for a competition-specific waiver.

---

## Comparison summary

| corpus | licence | NC clause | redistribution | derivative works | competition-use risk |
|---|---|---|---|---|---|
| MRNet | Stanford Research Use Agreement | yes | prohibited | prohibited | **high** |
| fastMRI | NYU data sharing agreement | yes (dataset only) | not explicitly prohibited | explicitly allowed for algorithms | **moderate** |
| fastMRI+ | Microsoft GitHub + fastMRI base | yes (base images) | not explicitly prohibited | explicitly allowed for algorithms | **moderate** |

---

## Recommended next actions

1. **MRNet:** Do not include in a Kaggle submission without host approval or a written licence exception. Update `knee-ombx` and `knee-lkni` with this finding.
2. **fastMRI:** Proceed with algorithm-use only; do not redistribute raw data in public Kaggle datasets; disclose as external data in submission metadata.
3. **fastMRI+:** Confirm the annotation CSV licence in `microsoft/fastmri-plus`; if it is MIT/Apache/CC-BY, the only remaining barrier is the base fastMRI NC clause.
4. **Documentation:** Add this audit to `comp/members.declines` reasoning if MRNet was under consideration for a member.

## References

- `comp/COMPETITION_RULES.txt`, Section 2.6
- MRNet Research Use Agreement: [stanfordmlgroup.github.io commit `a07223d`](https://github.com/stanfordmlgroup/stanfordmlgroup.github.io/commit/a07223dfda18ebae2a09a8c43e11e7365c3ccc90.patch)
- fastMRI paper: Knoll et al., "fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning" ([PMC6996599](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/))
- fastMRI+ repo: [microsoft/fastmri-plus](https://github.com/microsoft/fastmri-plus)
- Existing project audit: `research/deep_dives/2026-09-02_external_knee_mri_data_access_audit.md`
