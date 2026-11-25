# F1 drawing plan

## Exemplars

The corpus query combined stage pipeline and model anatomy with the vision and science/engineering subfields, missing joints, temporal interpolation, and two experts.

- [SceneMI, ICCV 2025, Figure 1](https://openaccess.thecvf.com/content/ICCV2025/papers/Hwang_SceneMI_Motion_In-betweening_for_Modeling_Human-Scene_Interaction_ICCV_2025_paper.pdf) shows how a consistent pose sequence lets readers distinguish observations and completions.
- [SILK, CVPR 2025 Workshop, Figure 1](https://openaccess.thecvf.com/content/CVPR2025W/HuMoGen/papers/Akhoundi_SILK_Smooth_InterpoLation_frameworK_for_motion_in-betweening_CVPRW_2025_paper.pdf) gives the poses visual prominence and leaves explanations to the caption.
- [Diffusion Priors for Ill-Posed Skeleton Reconstruction, Mathematics, Figure 1](https://www.mdpi.com/2227-7390/14/18/3317) uses skeletal line art to show a missing limb at print size.
- [OFPI, Mathematics, Figure 1](https://www.mdpi.com/2227-7390/11/6/1451) frames a skeletal pipeline around recognizable visual inputs and outputs.
- FigGenie corpus iclr25-0056_fig1 (DiffPuter) follows the same synthetic input through its changing stages. Keep the same pose endpoints and change only the missing part between strips.
- FigGenie corpus cvpr24-0031_fig1 (IReNe) opens up a new mechanism in the flow. Give the kernel bands and endpoint projection their own glyphs.
- FigGenie corpus icml23-0028_fig1 (ProtST) makes branch-level hierarchy legible with a compact secondary path.

The poses, kernel glyphs and routes were drawn for this figure. Exemplar images and parts stayed in the reference set.

## Canvas and layout

Use the Mathematics full-span width in the MDPI adjustwidth environment, 184.7 mm (523.559 PDF points), with a 150 pt figure height. The class width probe reports 394.35522 TeX pt for the body and 525.52245 TeX pt in adjustwidth. The conversion to PDF points uses 72/72.27. Place the input and output strips on the same baseline. Keep six operation blocks in a horizontal row and route the short-gap branch under the main flow. Use 6.5 pt Arimo labels and bold on the two highlighted contributions.

The main path progresses from the sparse sequence through interpolation, learnable kernel features, the input bridge, the long-gap TCN and the output projection, then through the router to the completed sequence. The short-gap expert shares the interpolation starting point and joins at the router. The full visibility-mask input and addition of the corrected residual to the interpolant are detailed in the methods text.

The two highlighted operations use muted teal (#216D72 on #E3F0EE). Neutral operations and observed poses use gray. Preserve hierarchy in grayscale with the striped kernel tap glyph, the constrained residual meeting two endpoint dots, and the smaller short-gap branch. The projected residual is solid and the raw residual is dashed.

## Review at print size

- Compare the pose strips at the same scale and check that only the previously missing limb changes.
- Check that the long and short paths both end at the duration router.
- Check that labels fit and arrowheads stay clear of them.
- Confirm that the projected residual meets both endpoint dots.
- Check both novelty tiles in grayscale and color.
- Validate the SVG with FigGenie before exporting PDF and editable PowerPoint.

FigGenie lint's sparse-text advisory follows the brief's text limit. The short-gap TCN label is centered beneath its glyph.