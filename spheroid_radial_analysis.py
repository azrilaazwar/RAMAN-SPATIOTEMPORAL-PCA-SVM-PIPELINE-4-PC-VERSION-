"""
spheroid_radial_analysis.py
============================
Automated equatorial-plane selection, spheroid segmentation, and radial
intensity profiling for confocal z-stack immunofluorescence images of
3D spheroids.

Pipeline
--------
1. Load a single-channel confocal z-stack (nuclei channel).
2. Identify the true equatorial (widest) optical plane automatically,
   using a threshold computed once from the highest-contrast slice and
   applied uniformly across the stack (avoids threshold instability at
   low-contrast, out-of-focus planes).
3. Segment the whole-spheroid boundary on that plane (Otsu thresholding
   + morphological cleanup + largest-connected-component selection).
4. Compute per-pixel normalized radial distance from the spheroid
   centroid (0 = center, 1 = surface).
5. Apply the identical mask/geometry (derived from the nuclei channel)
   to co-registered marker channels for the same spheroid.
6. Bin marker intensity into concentric radial shells (radial profile),
   and/or compute a simple Core (inner X%) vs Whole intensity ratio.
7. Aggregate across biological replicates (spheroids) and run
   between-condition statistics (Welch's t-test, 95% CI via
   t-distribution) at the spheroid level -- never pooling pixels
   across replicates.

Expected input
--------------
Single-channel, single-fluorophore TIFF z-stacks (uint16), one file per
channel per spheroid, e.g.:
    A2058_CAIX_TF1_S1C1.tif   (condition=A2058, sample=S1, channel=1)
    A2058_CAIX_TF1_S1C2.tif   (condition=A2058, sample=S1, channel=2)
Exported from Fiji via Image > Color > Split Channels, saved directly
(not flattened to RGB, not exported as a rendered snapshot -- rendered/
display-export formats bake in brightness/contrast adjustments and are
unsuitable for quantification).

Dependencies
------------
numpy, scipy, scikit-image, pandas, matplotlib, tifffile

    pip install numpy scipy scikit-image pandas matplotlib tifffile

Author: [your name]
License: [your choice, e.g. MIT]
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.stats import binned_statistic, ttest_ind
from scipy.stats import t as tdist
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------

@dataclass
class SpheroidGeometry:
    whole_mask: np.ndarray
    core_mask: np.ndarray
    centroid_y: float
    centroid_x: float
    r_equiv: float
    normalized_radius: np.ndarray  # same shape as image; only valid where whole_mask is True


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def remove_border_artifact(img: np.ndarray) -> np.ndarray:
    """
    Zero out a solid, uniform-value frame around the image edge if present
    (e.g. a UI/window artifact from a non-calibrated export). Only
    triggers if >90% of a 15-px border strip is one dominant value > 5;
    real biological images essentially never satisfy this, so it is a
    safe no-op on clean data.
    """
    strip = np.concatenate([
        img[0:15, :].ravel(), img[-15:, :].ravel(),
        img[:, 0:15].ravel(), img[:, -15:].ravel(),
    ])
    vals, counts = np.unique(strip, return_counts=True)
    dominant_val = vals[np.argmax(counts)]
    dominant_frac = counts.max() / strip.size
    if dominant_val > 5 and dominant_frac > 0.9:
        img = img.copy()
        img[img == dominant_val] = 0
    return img


def subtract_baseline(img: np.ndarray) -> np.ndarray:
    """
    Subtract a per-image fixed baseline (detector offset / dark current),
    estimated as the 1st percentile pixel value. Do this before any
    downstream segmentation or intensity measurement.
    """
    img = img.astype(float)
    return img - np.percentile(img, 1)


# --------------------------------------------------------------------------
# Equatorial plane selection
# --------------------------------------------------------------------------

def _area_at_fixed_threshold(img2d: np.ndarray, thr: float) -> int:
    """Largest connected-component area in img2d after smoothing + fixed threshold."""
    sm = gaussian(img2d.astype(float), sigma=4)
    mask = closing(sm > thr, disk(4))
    mask = remove_small_objects(mask, min_size=500)
    lbl = label(mask)
    if lbl.max() == 0:
        return 0
    return max(p.area for p in regionprops(lbl))


def find_equatorial_slice(stack: np.ndarray):
    """
    Identify the widest (equatorial) optical plane in a z-stack.

    Threshold is computed ONCE, from the slice with the highest 99th-
    percentile intensity (the most reliable, highest-contrast slice),
    then applied identically to every slice. This matters: naively
    recomputing Otsu on every individual slice fails at deeper,
    lower-contrast planes -- Otsu's threshold estimate can collapse on
    low-contrast slices, causing near-total-frame false segmentation.
    Using one threshold from a trustworthy reference slice avoids this.

    Parameters
    ----------
    stack : ndarray, shape (Z, Y, X)
        Nuclei-channel z-stack.

    Returns
    -------
    z_equatorial : int
        Index of the selected equatorial slice.
    areas : ndarray
        Raw cross-sectional area per z-slice.
    areas_smoothed : ndarray
        3-point moving-average smoothed version of `areas`.
    """
    p99_per_slice = [np.percentile(stack[z], 99) for z in range(stack.shape[0])]
    reference_z = int(np.argmax(p99_per_slice))

    smoothed_ref = gaussian(stack[reference_z].astype(float), sigma=4)
    fixed_threshold = threshold_otsu(smoothed_ref[smoothed_ref > 0]) * 0.5

    areas = np.array([_area_at_fixed_threshold(stack[z], fixed_threshold)
                       for z in range(stack.shape[0])])
    areas_smoothed = np.convolve(areas, np.ones(3) / 3, mode="same")
    z_equatorial = int(np.argmax(areas_smoothed))
    return z_equatorial, areas, areas_smoothed


# --------------------------------------------------------------------------
# Segmentation and geometry
# --------------------------------------------------------------------------

def segment_spheroid(img: np.ndarray, core_fraction: float = 0.4) -> SpheroidGeometry:
    """
    Segment the whole-spheroid boundary on a single 2D plane (already
    background-subtracted) and compute normalized radial geometry.

    Parameters
    ----------
    img : ndarray, shape (Y, X)
        Background-subtracted nuclei-channel plane.
    core_fraction : float
        Core ROI radius as a fraction of the equivalent whole-spheroid
        radius (e.g. 0.4 = innermost 40% of the radius).

    Returns
    -------
    SpheroidGeometry
    """
    smoothed = gaussian(img, sigma=4)
    nonzero = smoothed[smoothed > 0]
    threshold = threshold_otsu(nonzero) if nonzero.size else threshold_otsu(smoothed)

    mask = closing(smoothed > threshold * 0.5, disk(5))
    mask = remove_small_objects(mask, min_size=800)
    labeled = label(mask)
    if labeled.max() == 0:
        raise RuntimeError("No spheroid detected -- check image / threshold.")

    props = regionprops(labeled, intensity_image=img)
    largest = max(props, key=lambda p: p.area)
    whole_mask = ndi.binary_fill_holes(labeled == largest.label)

    cy, cx = largest.centroid
    r_equiv = np.sqrt(whole_mask.sum() / np.pi)

    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    normalized_radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r_equiv
    core_mask = (normalized_radius <= core_fraction) & whole_mask

    return SpheroidGeometry(
        whole_mask=whole_mask, core_mask=core_mask,
        centroid_y=cy, centroid_x=cx, r_equiv=r_equiv,
        normalized_radius=normalized_radius,
    )


# --------------------------------------------------------------------------
# Intensity quantification
# --------------------------------------------------------------------------

def core_whole_ratio(img: np.ndarray, geometry: SpheroidGeometry) -> dict:
    """Mean intensity in Whole and Core ROIs, and their ratio, for one channel."""
    whole_mean = img[geometry.whole_mask].mean()
    core_mean = img[geometry.core_mask].mean()
    return dict(whole_mean=whole_mean, core_mean=core_mean, ratio=core_mean / whole_mean)


def radial_profile(img: np.ndarray, geometry: SpheroidGeometry, n_bins: int = 20) -> np.ndarray:
    """
    Mean intensity per concentric radial bin (0=center to ~1=surface),
    computed for ONE spheroid/channel. Aggregate across spheroids
    afterward -- do not pool pixels across replicates.
    """
    bin_edges = np.linspace(0, 1.05, n_bins + 1)
    d = geometry.normalized_radius[geometry.whole_mask]
    v = img[geometry.whole_mask]
    means, _, _ = binned_statistic(d, v, statistic="mean", bins=bin_edges)
    return means


def bin_centers(n_bins: int = 20) -> np.ndarray:
    edges = np.linspace(0, 1.05, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


# --------------------------------------------------------------------------
# Statistics (spheroid-level, never pixel-pooled)
# --------------------------------------------------------------------------

def mean_and_ci95(profiles):
    """
    Mean and 95% CI across biological replicates (one profile array per
    spheroid, all the same length). Uses the t-distribution at
    df = n_replicates - 1, appropriate for small n.
    """
    arr = np.array(profiles)
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1)
    t_crit = tdist.ppf(0.975, df=n - 1)
    ci95 = t_crit * sd / np.sqrt(n)
    return mean, ci95


def compare_conditions(values_a: pd.Series, values_b: pd.Series) -> dict:
    """Welch's unpaired t-test between two conditions' spheroid-level values."""
    t_stat, p_value = ttest_ind(values_a, values_b, equal_var=False)
    return dict(
        mean_a=values_a.mean(), sd_a=values_a.std(),
        mean_b=values_b.mean(), sd_b=values_b.std(),
        t=t_stat, p=p_value,
    )


# --------------------------------------------------------------------------
# End-to-end example
# --------------------------------------------------------------------------

def analyze_spheroid(nuclei_stack_path: str, marker_stack_paths: dict,
                      core_fraction: float = 0.4, n_bins: int = 20) -> dict:
    """
    Full single-spheroid pipeline: find equatorial plane from the nuclei
    stack, segment it, then measure Core/Whole ratio and radial profile
    for the nuclei channel plus every marker channel provided (using the
    SAME geometry derived from nuclei -- markers are not independently
    thresholded).

    Parameters
    ----------
    nuclei_stack_path : str
        Path to the nuclei-channel z-stack TIFF.
    marker_stack_paths : dict[str, str]
        Mapping of marker name -> z-stack TIFF path, e.g.
        {"CA IX": "...C2.tif", "COUP-TFI": "...C3.tif"}.

    Returns
    -------
    dict with keys: z_equatorial, areas, areas_smoothed, geometry, results
        results is {channel_name: {"core_whole": {...}, "radial_profile": ndarray}}
    """
    nuclei_stack = tifffile.imread(nuclei_stack_path)
    z_eq, areas, areas_smoothed = find_equatorial_slice(nuclei_stack)

    nuclei_img = subtract_baseline(remove_border_artifact(nuclei_stack[z_eq]))
    geometry = segment_spheroid(nuclei_img, core_fraction=core_fraction)

    results = {
        "Nuclei": {
            "core_whole": core_whole_ratio(nuclei_img, geometry),
            "radial_profile": radial_profile(nuclei_img, geometry, n_bins=n_bins),
        }
    }
    for marker_name, path in marker_stack_paths.items():
        stack = tifffile.imread(path)
        img = subtract_baseline(remove_border_artifact(stack[z_eq]))  # SAME z-index as nuclei
        results[marker_name] = {
            "core_whole": core_whole_ratio(img, geometry),
            "radial_profile": radial_profile(img, geometry, n_bins=n_bins),
        }

    return dict(z_equatorial=z_eq, areas=areas, areas_smoothed=areas_smoothed,
                geometry=geometry, results=results)


if __name__ == "__main__":
    # --- Example usage / template -- edit paths and conditions for your data ---
    example = analyze_spheroid(
        nuclei_stack_path="data/A2058_CAIX_TF1_S1C1.tif",
        marker_stack_paths={
            "CA IX": "data/A2058_CAIX_TF1_S1C2.tif",
            "COUP-TFI": "data/A2058_CAIX_TF1_S1C3.tif",
        },
    )
    print(f"Equatorial plane: z={example['z_equatorial']}")
    for channel, res in example["results"].items():
        cw = res["core_whole"]
        print(f"{channel}: Core/Whole ratio = {cw['ratio']:.3f} "
              f"(Core mean={cw['core_mean']:.1f}, Whole mean={cw['whole_mean']:.1f})")

    # To compare two conditions across n spheroids: run analyze_spheroid() for
    # each spheroid, collect the per-spheroid ratio (or radial_profile) values
    # into two lists (one per condition), then:
    #
    #   from spheroid_radial_analysis import compare_conditions
    #   import pandas as pd
    #   result = compare_conditions(pd.Series(condition_a_values), pd.Series(condition_b_values))
    #   print(result)
