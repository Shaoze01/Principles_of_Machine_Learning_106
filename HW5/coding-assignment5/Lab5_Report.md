#  Report: Restoring Clean Images from Noisy Images Using Pairwise MRF

## (a) Paper Design

The image denoising problem is modeled as a pairwise Markov Random Field (MRF) with two layers: an observed noisy layer $y_i \in \{-1, +1\}$ and a hidden clean layer $x_i \in \{-1, +1\}$. The energy function is:

$$E(\vec{x}, \vec{y}) = h\sum_i x_i - \beta \sum_{\{i,j\}} x_ix_j - \eta\sum_i x_iy_i$$

where $\{i,j\}$ denotes 4-connected neighboring pixel pairs. Each term serves a distinct role:

- **$h\sum_i x_i$**: Bias term, penalizing a global preference toward $+1$ or $-1$.
- **$-\beta\sum x_ix_j$**: Smoothness prior, encouraging neighboring pixels to share the same sign.
- **$-\eta\sum x_iy_i$**: Data fidelity, encouraging each hidden pixel to agree with its observed value.

The optimization uses a coordinate-descent (ICM) algorithm: initialize $x_i = y_i$, then sweep over all pixels, flipping $x_i \to -x_i$ whenever the local energy change $\Delta E = -2x_i(h - \beta\sum_{j \in \mathcal{N}(i)} x_j - \eta y_i) < 0$. Repeat until no pixel is flipped.

## (b) Model Description and Parameter Justification

### Preprocessing: From RGB to Binary

The original images are color. A critical preprocessing step is converting them to binary $\{-1, +1\}$ arrays. Two approaches were explored:

1. **`PIL convert(mode='1', dither=0)`**: 

2. **RGB average with adjusted threshold (adopted approach)**: 

### MRF Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| $h$ | 0 | The image contains substantial regions of both black and white. No global bias is needed. |
| $\beta$ | 2 | A moderately strong smoothness prior. Encourages neighboring pixels to agree, which effectively corrects isolated noisy pixels that disagree with all their neighbors. |
| $\eta$ | 1 | Since ~10% of pixels are corrupted, most observed values are correct. $\eta = 1$ provides a baseline trust in the data. |



## (c) Evaluation

With $h = 0$, $\beta = 2$, $\eta = 1$, and the RGB-average preprocessing (threshold = 95):

| Metric | Value |
|--------|-------|
| Image size | 654 × 489 (319,806 pixels) |
| Initial accuracy | 91.80% |
| Final recovery rate | **96.09%** |
| Iterations to converge | 8 |

Iteration-by-iteration convergence:

| Iteration | Pixels Flipped | Accuracy |
|-----------|---------------|----------|
| 1 | 14,074 | 95.96% |
| 2 | 413 | 96.06% |
| 3 | 88 | 96.08% |
| 4 | 18 | 96.09% |
| 5 | 6 | 96.09% |
| 6 | 3 | 96.09% |
| 7 | 1 | 96.09% |
| 8 | 0 (converged) | 96.09% |

The algorithm converges rapidly: the first iteration contributes the vast majority of corrections (from 91.80% to 95.96%), with subsequent iterations making diminishing refinements. The remaining ~4% errors are concentrated at edges and fine details where the smoothness prior conflicts with the true boundary structure — an inherent limitation of the pairwise MRF model.

## (d) Reflections

This lab highlighted that **preprocessing choices can be just as critical as model parameters**. The `convert('1')` approach seemed like a natural default, but its internal luminance weighting was poorly suited for this yellow-blue image. whatever MRF parameter are choosed, the results were not satisfactory. Switching to a simple RGB average with a carefully chosen threshold immediately resolved the issue. This reinforced the importance of examining raw data and sanity-checking every pipeline stage\.

On the MRF side, the key insight is the interplay between $\beta$ (smoothness) and $\eta$ (data fidelity): the ratio $\beta / \eta$ determines whether the neighborhood can outvote a noisy observation. Coordinate-descent (ICM) is simple and fast, but it only finds a local minimum. 
