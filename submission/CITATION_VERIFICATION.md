# Citation Verification Report

Manuscript: *Endpoint-Constrained Multi-Band Fractional Memory for Repairing Long-Duration Limb Occlusions in Dance Motion Capture*

Audit date: 3 September 2026

## Outcome

- All 23 cited works were located in an official publisher, proceedings, repository, or DOI record.
- Every BibTeX key is cited in the manuscript, and every manuscript citation resolves to one BibTeX entry.
- No fabricated, duplicate, or unresolved reference was found.
- Two bibliographic records required correction: `jiang2022dmae` was upgraded from its arXiv record to the ACM Multimedia 2022 version of record; `carissimi2018gaps` was corrected to the 2019 Springer proceedings publication.
- Four sentences were narrowed or repositioned so that the cited source supports the exact statement made.

## Entry-by-Entry Verification

| Key | Verified record | Citation role in manuscript | Status |
|---|---|---|---|
| `li2021aist` | Li et al., ICCV 2021, pp. 13401-13412 | AIST++ dataset, dance generation, 1,408 sequences and 10 genres | Verified |
| `kim2022dance` | Kim et al., CVPR 2022, pp. 3490-3500 | Music-conditioned generation controlled by multiple dance genres | Verified |
| `siyao2022bailando` | Li Siyao et al., CVPR 2022, pp. 11050-11059 | Actor-critic GPT and choreographic memory | Verified |
| `tseng2023edge` | Tseng et al., CVPR 2023, pp. 448-458 | Editable diffusion-based dance generation | Verified |
| `li2023finedance` | Li et al., ICCV 2023, pp. 10234-10243 | Fine-grained full-body dance dataset and additional genres | Verified |
| `hu2021dance` | Hu and Ahuja, ICCV 2021, pp. 11015-11024 | Hierarchical 3D-pose-based dance recognition | Verified |
| `cui2021mtgcn` | Cui and Sun, CVPR 2021, pp. 4801-4810 | Motion prediction from incomplete observations with auxiliary repair | Verified |
| `duan2022completion` | Duan et al., AAAI 2022, 36(4), pp. 4459-4467, DOI 10.1609/aaai.v36i4.20368 | Unified in-betweening, infilling, and blending | Verified |
| `jiang2022dmae` | Jiang et al., ACM Multimedia 2022, pp. 5123-5131, DOI 10.1145/3503161.3547796 | Dual masking of spatial and temporal skeleton tokens for completion | Corrected and verified |
| `mascaro2024unimask` | Valls Mascaro et al., AAAI 2024, 38(6), pp. 5261-5269, DOI 10.1609/aaai.v38i6.28333 | Patchified skeleton masked autoencoder for motion synthesis | Verified |
| `wang2024sic` | Wang et al., CVPR 2024, pp. 2436-2446 | Unified skeleton sequence tasks through in-context modeling | Verified |
| `shao2026finetec` | Shao et al., AAAI 2026, 40(11), pp. 8842-8850, DOI 10.1609/aaai.v40i11.37838 | Completion and recognition under temporal corruption | Verified |
| `xia2016recovery` | Xia et al., *Information Sciences* 339, pp. 189-205, DOI 10.1016/j.ins.2015.12.041 | Statistical and kinematic motion recovery | Verified |
| `carissimi2018gaps` | Carissimi et al., *Computer Vision - ECCV 2018 Workshops*, LNCS 11130, pp. 364-379, Springer 2019, DOI 10.1007/978-3-030-11012-3_29 | Missing-joint prediction with denoising autoencoders | Corrected and verified |
| `hernandez2019inpainting` | Hernandez et al., ICCV 2019, pp. 7134-7143 | Spatio-temporal inpainting formulation for motion prediction | Verified |
| `lohit2021latent` | Lohit et al., WACV 2021, pp. 2342-2351 | Recovery of unmarked joints through latent-space optimization | Verified |
| `kim2017tcn` | Kim and Reiter, CVPR Workshops 2017, pp. 20-28 | Residual temporal convolution for 3D skeleton sequences | Verified |
| `loshchilov2019adamw` | Loshchilov and Hutter, ICLR 2019 | AdamW optimizer | Verified |
| `efron1994bootstrap` | Efron and Tibshirani, *An Introduction to the Bootstrap*, DOI 10.1201/9780429246593 | Percentile bootstrap confidence intervals | Verified |
| `podlubny1999fractional` | Podlubny, *Fractional Differential Equations*, Academic Press, 1999, ISBN 978-0-12-558840-9 | GL operators, algebraic memory kernels, and fractional-calculus foundations | Verified |
| `diethelm2010analysis` | Diethelm, *The Analysis of Fractional Differential Equations*, LNM 2004, Springer 2010, DOI 10.1007/978-3-642-14574-2 | Fractional differential-equation foundations | Verified |
| `wang2025pose3dm` | Wang et al., *Fractal and Fractional* 9(9), 603, DOI 10.3390/fractalfract9090603 | Fractional total-variation regularization for temporally stable 3D pose estimation | Verified |
| `guo2026fotsr` | Guo et al., *Fractal and Fractional* 10(6), 420, DOI 10.3390/fractalfract10060420 | GL fractional differential loss for human-motion prediction | Verified |

## Textual Corrections Made

1. The opening sentence now attributes only dance generation and recognition to AIST++ and HDVR; unsupported mention of motion retrieval was removed.
2. The recovery survey now says "motion recovery and related completion or prediction problems" because the cited inpainting work is principally a prediction paper.
3. The TCN citation now explicitly refers to residual temporal modeling for 3D skeleton sequences rather than implying that Kim and Reiter originated every implementation detail of the present TCN.
4. The bootstrap citation now follows only the bootstrap confidence interval clause; paired Cohen's d is no longer implicitly attributed to the bootstrap monograph.

## Primary Verification Sources

- Computer Vision Foundation Open Access: https://openaccess.thecvf.com/
- AAAI proceedings: https://ojs.aaai.org/index.php/AAAI
- ACM DOI record: https://doi.org/10.1145/3503161.3547796
- Springer DOI record: https://doi.org/10.1007/978-3-030-11012-3_29
- ScienceDirect record: https://doi.org/10.1016/j.ins.2015.12.041
- MDPI records: https://doi.org/10.3390/fractalfract9090603 and https://doi.org/10.3390/fractalfract10060420
- Springer monograph: https://doi.org/10.1007/978-3-642-14574-2
- Taylor and Francis monograph: https://doi.org/10.1201/9780429246593
- Elsevier monograph: ISBN 978-0-12-558840-9

## Automated Check Reconciliation

The online verifier reached 15 of the 23 locators successfully. Eight requests returned CVF connection resets or DOI endpoint HTTP 403/503 responses. These are access failures, not bibliographic mismatches. The eight affected entries were resolved manually from official proceedings, publisher pages, or Crossref metadata; the unmodified automatic report and the manual-resolution addendum are retained in the AutoWriting audit directory.
