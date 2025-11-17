import numpy as np
import cv2
import os
from scipy import stats
from scipy.spatial.distance import pdist


###################################################################################################
# Common helper functions
###################################################################################################

def preprocess_image(img_path):
    """
    Common preprocessing steps for all contour operations.
    
    Args:
        img_path (str): Path to input image
    Returns:
        tuple: (original image, edges, contours)
    """

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, edges, contours

def get_contours(img_path):
    """
    Helper function to get contours from image path.
    
    Args:
        img_path (str): Path to input image
    Returns:
        list: List of contours
    """
    _, _, contours = preprocess_image(img_path)
    return contours

def get_contour_image(img_path):
    """
    Generate a contour image from input image path.
    
    Args:
        img_path (str): Path to input image
    Returns:
        np.ndarray: Binary image showing detected contours
    """
    img, _, contours = preprocess_image(img_path)
    contour_img = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cv2.drawContours(contour_img, contours, -1, (255,255,255), 1)
    return contour_img


###################################################################################################
# Contour Features Calculations
###################################################################################################

def get_contour_lengths(img_path, min_length=10):
    """
    Calculate lengths of all contours in an image.
    
    Args:
        img_path (str): Path to input image
        min_length (float): Minimum contour length to include
    Returns:
        np.ndarray: Array of contour lengths
    """
    _, _, contours = preprocess_image(img_path)
    lengths = np.array([cv2.arcLength(cnt, True) for cnt in contours])
    # Filter out very small contours that might be noise
    return lengths[lengths >= min_length]

def get_contour_orientations(img_path, min_points=5):
    """
    Calculate orientations of all contours in an image.
    
    Args:
        img_path (str): Path to input image
        min_points (int): Minimum points needed in contour for orientation calculation
    Returns:
        np.ndarray: Array of contour orientations in degrees (0-179)
    """
    _, _, contours = preprocess_image(img_path)
    orientations = []
    
    for cnt in contours:
        if len(cnt) >= min_points:
            try:
                # Fit an ellipse to the contour
                (_, _), (MA, ma), angle = cv2.fitEllipse(cnt)
                # Normalize angle to 0-179 range (since orientation has 180-degree symmetry)
                angle = angle % 180
                orientations.append(angle)
            except cv2.error:
                # Skip contours that can't be fit with an ellipse
                continue
    
    return np.array(orientations)

def junction_density(img_path, grid_size=20, distance_threshold=3):
    """
    Calculate approximate junction density using grid-based sampling.
    Based on Wilder et al. (2018) & simplified for computational efficiency.
    
    Args:
        img_path: Path to input image
        grid_size: Size of grid cells for sampling (larger = faster but coarser)
        distance_threshold: Distance to consider as junction
    Returns:
        float: Junction density score
    """
    _, _, contours = preprocess_image(img_path)
    if not contours:
        return 0.0
    
    # Create grid cells
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Sample points along contours at regular intervals
    sampled_points = []
    for cnt in contours:
        # Reduce points to a fixed number per contour
        step = max(len(cnt) // 20, 1)  # Sample ~20 points per contour
        points = cnt.reshape(-1, 2)[::step]
        sampled_points.extend(points)
    
    sampled_points = np.array(sampled_points)
    if len(sampled_points) < 2:
        return 0.0
    
    # Divide image into grid cells and count junctions in each cell
    junction_count = 0
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            # Get points in this grid cell
            mask = (sampled_points[:, 0] >= j) & (sampled_points[:, 0] < j + grid_size) & \
                   (sampled_points[:, 1] >= i) & (sampled_points[:, 1] < i + grid_size)
            points_in_cell = sampled_points[mask]
            
            # If we have multiple points in cell, check for junctions
            if len(points_in_cell) > 1:
                # Simple clustering - if points are close, count as junction
                
                if len(pdist(points_in_cell)) > 0:
                    distances = pdist(points_in_cell)
                    junction_count += np.sum(distances < distance_threshold)
    
    # Normalize by image area (per million pixels as before)
    density = junction_count / ((h * w) / 1000000)
    
    return density


def parallelism(img_path, angle_threshold=10, dist_threshold=20):
    """
    Calculate global parallelism score.
    Based on Rezanejad et al. (2024).
    
    Args:
        img_path: Path to input image
        angle_threshold: Maximum angle difference to consider parallel
        dist_threshold: Maximum distance to consider parallel relationship
    Returns:
        float: Parallelism score between 0 and 1
    """
    _, _, contours = preprocess_image(img_path)
    if not contours:
        return 0.0
    
    parallel_segments = 0
    total_comparisons = 0
    
    # Compare each contour with every other contour
    for i, cnt1 in enumerate(contours):
        # Get orientation of first contour
        if len(cnt1) < 5:
            continue
        try:
            _, _, angle1 = cv2.fitEllipse(cnt1)
        except:
            continue
            
        for cnt2 in contours[i+1:]:
            if len(cnt2) < 5:
                continue
            try:
                _, _, angle2 = cv2.fitEllipse(cnt2)
                
                # Check if angles are parallel (considering 180 degree symmetry)
                angle_diff = min(abs(angle1 - angle2) % 180, abs(180 - (abs(angle1 - angle2) % 180)))
                
                if angle_diff < angle_threshold:
                    parallel_segments += 1
                total_comparisons += 1
                    
            except:
                continue
    
    return parallel_segments / max(total_comparisons, 1)


def get_contour_stats(img_path):
    """
    Get comprehensive contour statistics for an image.
    
    Args:
        img_path (str): Path to input image
    Returns:
        dict: Dictionary containing various contour statistics
    """
    lengths = get_contour_lengths(img_path)
    orientations = get_contour_orientations(img_path)
    
    stats = {
        'num_contours': len(lengths),
        'mean_length': np.mean(lengths) if len(lengths) > 0 else 0,
        'std_length': np.std(lengths) if len(lengths) > 0 else 0,
        'median_length': np.median(lengths) if len(lengths) > 0 else 0,
        'orientation_hist': np.histogram(orientations, bins=18, range=(0,180))[0] if len(orientations) > 0 else np.zeros(18),
        'junction_density': junction_density(img_path),
        'parallelism_score': parallelism(img_path),
    }
    
    return stats
