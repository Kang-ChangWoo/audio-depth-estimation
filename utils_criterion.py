
import numpy as np 


def compute_errors(gt, pred, min_depth_threshold=0.0):
    """Computation of error metrics between predicted and ground truth depths
        Taken from Beyond Image to Depth Github repository
    
    Args:
        gt: Ground truth depth map
        pred: Predicted depth map
        min_depth_threshold: Minimum depth threshold to exclude near-depth pixels (default: 0.0)
                            Set to 0.0 to include all valid pixels, or 0.1 to match training threshold
    """
    # Use threshold to filter pixels (default: 0.0 to include all valid pixels)
    # mask = gt > min_depth_threshold
    mask = gt != 0.0
    if mask.sum() == 0:
        # If no valid GT pixels, return zeros
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    pred = pred[mask]
    gt = gt[mask]
    
    # Additional check: filter out very small predictions to avoid division by zero
    # Use a more lenient threshold (1e-3 for depth in meters, or 1e-6 for normalized)
    # Since depth can be in meters (0-30m range), use 1e-3 as threshold
    epsilon = 1e-3 if gt.max() > 1.0 else 1e-6  # Adaptive threshold based on scale
    valid_mask = (pred > epsilon) & (gt > epsilon)
    
    if valid_mask.sum() == 0:
        # If no valid pixels after filtering, this means all predictions are invalid
        # (either negative or too small). Only use pixels where GT is valid,
        # and explicitly exclude negative predictions to prevent them from affecting ABS_REL
        valid_mask = gt > epsilon
        if valid_mask.sum() == 0:
            # No valid GT pixels either, return zeros
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Filter out negative predictions explicitly
        valid_mask = valid_mask & (pred > 0)  # Exclude negative predictions
        if valid_mask.sum() == 0:
            # All predictions are negative or zero, return high error to indicate failure
            # This prevents negative predictions from being included in ABS_REL calculation
            print(f"WARNING: All predictions are negative or zero. "
                  f"GT range: [{gt.min():.3f}, {gt.max():.3f}], "
                  f"Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
            return 1.0, gt.max(), 0.0, 0.0, 0.0, 1.0, gt.max()
    
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    # Avoid division by zero - use epsilon based on data scale
    epsilon = 1e-3 if gt.max() > 1.0 else 1e-6
    thresh = np.maximum((gt / np.maximum(pred, epsilon)), (np.maximum(pred, epsilon) / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse or rmse == np.inf:
        rmse = 0.0
    if a1 != a1 or a1 == np.inf:
        a1=0.0
    if a2 != a2 or a2 == np.inf:
        a2=0.0
    if a3 != a3 or a3 == np.inf:
        a3=0.0
    
    # Avoid division by zero in abs_rel - use epsilon based on data scale
    epsilon = 1e-3 if gt.max() > 1.0 else 1e-6
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    # Avoid log of zero
    log_10 = (np.abs(np.log10(np.maximum(gt, epsilon))-np.log10(np.maximum(pred, epsilon)))).mean()
    mae = (np.abs(gt-pred)).mean()
    if abs_rel != abs_rel or abs_rel == np.inf:
        abs_rel=0.0
    if log_10 != log_10 or log_10 == np.inf:
        log_10=0.0
    if mae != mae or mae == np.inf:
        mae=0.0
    
    return abs_rel, rmse, a1, a2, a3, log_10, mae
