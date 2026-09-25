# Findings from `test2.pdf`

The fixture was a two-page maths PDF, SHA-256
`fa7e2312819bcf0d536cd3d2008d21d1a740e62c7452cea126743b65b41a37c0`.
These were single CPU runs and manual checks against the source, not accuracy
scores or stable throughput measurements. The benchmark scripts and generated
artifacts were removed after this review; the figures below are the retained
record, not an executable benchmark suite.

## Whole-page comparison

| Configuration | Load / download | Processing | Processing per page | Main observation |
| --- | ---: | ---: | ---: | --- |
| Paddle A: layout and OCR; tables, formulas, unwarping, orientation, charts and seals off | 9.031 s | 25.085 s | 12.542 s | Lost some equations, answer options and premises; OCR included watermark text. |
| Paddle B: A with tables and formulas on | 12.951 s | 341.558 s | 170.779 s | Recovered more mathematical structure, including Q14's determinant, but still lost some options and conditions. |
| Docling C: native PDF text, table structure and formula enrichment on; OCR off | 101.099 s | 1.434 s | 0.717 s | Exported little beyond the date heading and image placeholders; the short conversion time does not mean successful extraction. |

Paddle used cached models. Docling's initialization included downloads. The
processing figure includes first inference but excludes export. Paddle B cost
13.62 times A's processing time. Because tables and formulas changed together,
this does not isolate their costs. The fixture has no ordinary data table, so it
does not validate table extraction. A determinant is a formula here.

## Focused layout and formula checks

`docling-parse` rendered both pages faithfully at scale 2, or 144 DPI, and
exposed native word cells. The standalone PP-DocLayout-M layout model at 0.5
found 9 formula regions on page 1 and 14 on page 2, but missed the reference
regions for Q7's fourth answer and Q14's determinant. PP-FormulaNet-S ran on
23 detected crops. It recovered some Q3 structure but damaged Q4, Q10 and Q17;
it never received Q14. The corrected two-page run measured 0.451 s for parsing
and rendering, 0.244 s for layout, and 5.477 s for formula inference. These
stage timings exclude some preparation and export, and the output was less
complete than Paddle B.

The final bounded check used PP-DocLayoutV2 and PP-FormulaNet_plus-S for Q3,
Q4, Q7, Q10, Q14 and Q17. A formula box had to cover at least half of each
reference rectangle. V2 found **none of the six** at confidence 0.5. The one
follow-up pass at 0.4 also found none. No formula recognizer ran on V2 output.
A separate manual-crop control gave plus-S the six formulas directly:

| Question | plus-S on manual crop |
| --- | --- |
| Q3, fractions and roots | Incorrect LaTeX |
| Q4, sec² and tan² | Lost sec² |
| Q7, answer `(-1,-1,1)` | Correct |
| Q10, ellipse denominators | Denominators retained, equality damaged |
| Q14, 3×3 determinant | Incorrect |
| Q17, twelfth root | Root indices damaged |

V2's load took 26.203 s including its first download; four page inferences
across the two thresholds took 6.866 s. Plus-S loaded in 26.540 s including
download and processed the six manual crops in 3.578 s. Those times do not
represent a working combined pipeline because V2 did not route these formulas.

## Native-text fallback and product decision

The native `docling-parse` layer contained 483 word cells on page 1 and 388 on
page 2, including the answer values `84`, `154` and `308`. PP-DocLayout-M TEXT
boxes omitted those three cells, even though the native layer had them. V2
classified all three as TEXT at both tested thresholds. Keeping only words
claimed by a layout box can therefore silently discard valid content. Native
text avoided the literal watermark OCR pollution seen in Paddle A; no
watermark-removal filter was applied.

The proposed common path is native PDF extraction with all native words
preserved, plus layout analysis and specialist recognition only for regions
that need it. The exact V2/plus-S combination did **not** pass the formula
quality gate on this fixture. Missing formula detection and bad recognition
must be reported separately, and source page images should remain available
when either fails. The one application smoke run preserved `84`, `154`, `308`
and Q6's digit condition, but V2 found no formula regions on these pages, so
plus-S was never invoked. There is no basis here for a table implementation;
choose a representative table page first.

The application prototype from that smoke run was subsequently reverted at
the user's request. This note preserves the experimental decision without
claiming that the pipeline is implemented or that this PDF is fully transcribed.

Primary references: [Docling parse](https://github.com/docling-project/docling-parse/blob/main/README.md),
[Paddle layout detection](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/layout_detection.en.md),
[Paddle formula recognition](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/formula_recognition.en.md).
