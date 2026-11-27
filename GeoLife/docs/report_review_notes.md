# Review notes for the supplied report draft

The accompanying `report_draft.pdf` is the unchanged PDF supplied on 2026-11-27. Its main RF/GRU grid, key noise-seed repetition and per-class results agree with A's aggregate tables. The speed-drift and speed-group permutation values also agree with B's published outputs. The following edits remain pending; uploading the draft does not mark them as completed.

1. Page 5, Figure 2: replace "48 held-out folds" with 54 LOUO folds. Only 48 users contribute windows to the common-support score; each model is trained on the other 53 reference users.
2. Page 5, GRU description: the mask excludes padded positions from global pooling. The recurrent network still processes the padded tensor, including the backward direction; the mask does not remove all padding effects.
3. Page 7, interaction figure: define `I(d,n)=F(d,n)-F(d,0)-F(0,n)+F(0,0)` and its signs. The colour scale should cover the largest displayed value. Some RF additive references are below zero, so a large positive contrast is not a causal recovery estimate.
4. Pages 9–10: add the four-condition, six-group permutation comparison or an appendix reference, including 10 repeats, the permutation seed and cross-condition pairing. The existing report shows only the speed group. Include or link B's scenario mapping within its stated limitations.
5. Pages 8 and 10: the key `ds10_n10` mean model gap is 2.594 percentage points, close to the clean gap of 2.614 points; this condition does not show an additional robustness margin of that magnitude.
6. Explain the preprocessing/filtering chain, the 120-second pilot choice, when settings were frozen, and the original-point noise pairing. Use the reproduction guide for environment versions, inputs and commands.
7. The course guide allows 6–10 pages of main text including figures and tables, excluding disclosure, references and appendices. The current Introduction-to-Conclusion spans pages 1–12 and needs compression.

Earlier XGBoost, CNN, SMOTE and t-SNE exploration can remain in historical material. Do not mix their older environment-specific scores into the frozen final grid or present planned augmentation/ensemble experiments as completed work.
