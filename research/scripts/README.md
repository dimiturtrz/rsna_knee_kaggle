# research/scripts

One-off analyses, each kept beside the `.out.txt` it printed so the number a research doc quotes can be
traced to the code that produced it. They are records, not tools: most hardcode the author's data root
(`D:/data/volumetric/mri/...`) and session scratch paths, and none is imported by the packages. Code
that is meant to run anywhere resolves its roots through `core.paths.DataRoot` and the gitignored
`paths.yaml` (shell: `experiments/roots.sh`).
