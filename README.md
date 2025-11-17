# Statistical Analysis of Scene Structure in Real vs Synthetic Images

This project investigates how **real photographs** differ from **AI-generated images** (Stable Diffusion v1.5) by analyzing their **geometric structure** and **frequency content** using classical computer vision methods. The goal is not to build a classifier, but to understand *where and how* generative models diverge from real-world scene statistics.

## Dataset

We study four scene categories:

- **Forests**
- **Highways**
- **Cockpits**
- **Bedrooms**

Real images come from:

- **Toronto Scenes** dataset [11]
- **SUN397** dataset [13]

Synthetic images (~1,000 per category) were generated with **Stable Diffusion v1.5** using prompt engineering (base scene description, lighting, key objects) plus negative prompts to avoid artifacts. All images were resized/filtered to a common resolution of **800×600**, converted to grayscale, normalized, and smoothed with a 5×5 Gaussian kernel.

## Methods

We use two complementary perspectives:

### 1. Geometric Analysis (Contour-Based)

Using OpenCV, we:

- Extract edges (Canny) and **external contours** (`findContours`).
- Compute **contour length statistics** (mean, median, std) as proxy for structural complexity.
- Estimate **contour orientations** via ellipse fitting (`fitEllipse`) and build orientation histograms.
- Measure **parallelism**: fraction of contour pairs whose orientations differ by < 10° and lie within a distance threshold.
- Estimate **junction density**: locations where multiple contours meet within a small radius, normalized by image size.

These features characterize scene structure in terms of shape complexity, directionality, and how often structures intersect or align.

### 2. Frequency Analysis (DCT-Based)

To analyze texture and structural regularity, we apply the **2D Discrete Cosine Transform (DCT)**:

\[
D(u,v) = \frac{1}{\sqrt{2N}} \sum_{x=0}^{N-1}\sum_{y=0}^{N-1}
I(x,y)\cos\left(\frac{\pi(2x+1)u}{2N}\right)\cos\left(\frac{\pi(2y+1)v}{2N}\right)
\]

We:

- Compute **average DCT spectra** for real vs synthetic images.
- Visualize **difference heatmaps** in the frequency plane.
- Compare **histograms of DCT coefficient magnitudes**.
- Aggregate energy in **low / mid / high** frequency bands to see how power is distributed.

This reveals whether generative models introduce periodic artifacts or distort the natural frequency distribution of scenes.

## Key Findings

### Geometric (Contour) Structure

- **Forests** show the **largest gap** between real and synthetic:
  - Synthetic forests tend to have **shorter contours** and **less variance**, hinting at oversimplified vegetation.
  - Junction densities are often **higher** in synthetic forests, suggesting overcompensated fine detail (e.g., too many small intersecting structures).
- **Highways** show **moderate differences**:
  - Real highways exhibit broader contour length distributions and higher junction densities.
  - Synthetic highways often have **lower parallelism scores** and fewer complex intersections, implying simplified infrastructure.
- **Cockpits** are **closest** between real and synthetic:
  - The strong geometric constraints of man-made interiors appear easier for the model to replicate.
- **Bedrooms** show an interesting flip:
  - Synthetic bedrooms often have **longer contours** and more clutter, possibly due to the model adding extra objects.
  - Junction densities in synthetic bedrooms are sharply peaked, suggesting “template-like” structures (e.g., many very similar bed layouts).

Overall, synthetic images frequently either **oversimplify** natural complexity or **overcomplicate** certain details, deviating from real structural statistics.

### Frequency Domain (DCT) Structure

- Difference heatmaps show that **synthetic images have stronger low-frequency coefficients**, emphasizing smoother, more homogeneous patterns.
- Real images tend to have more energy in **higher frequency regions**, reflecting natural fine details and sharper edges.
- Histograms of DCT coefficients:
  - Real images have a **higher probability of near-zero coefficients**, consistent with large uniform regions (e.g., sky, walls).
  - Synthetic images show **broader coefficient distributions**, suggesting added noise and less “sparse” frequency structure.
- Band-wise analysis (low/mid/high) indicates that synthetics often have **elevated energy across all bands**, while real images show a more natural, decaying spectrum.

These results suggest that current generative models **do not replicate the natural distribution of spatial frequencies** and instead produce images with more diffuse energy across the spectrum.

## Conclusion & Future Directions

This work shows that **classical computer vision tools**—contours, orientations, junctions, and DCT—can expose systematic differences between real and AI-generated images that aren’t obvious to the naked eye. Even using an older model (Stable Diffusion v1.5), we find:

- Geometric statistics (contours, parallelism, junctions) differ meaningfully, especially in **complex natural scenes**.
- Frequency statistics (DCT patterns) reveal **non-natural energy distributions** in synthetic images.

Limitations and next steps:

- This study used a single model (SD v1.5) and four scene categories; extending to modern models and more categories would make the conclusions stronger.
- More rigorous prompt protocols and statistical tests (e.g., distribution distances, hypothesis testing) could quantify differences more precisely.
- Future work could add **color statistics**, **texture descriptors**, curvature, symmetry, and other higher-order features, and track how these statistics evolve with newer generative models.
 
