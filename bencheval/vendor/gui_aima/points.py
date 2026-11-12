"""Turn a patch-level attention map into a click point (backbone-neutral)."""
import torch


def get_prediction_region_point(attn_scores, n_width, n_height, top_n=30, activation_threshold=0.3, return_all_regions=True, rect_center=False):
    """
    1. Select activated patches
    2. Divide connected patches into different regions
    3. Calculate the average activation value for each region
    4. Select the region with the highest average activation value
    5. Return the center point of that region as the final prediction point
    """

    # Get patches with activation values greater than a certain proportion of the maximum activation value as activated patches
    # Get the highest activation value and threshold
    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    # Select all patches above the threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices
    
    # Convert indices to 2D coordinates
    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))
    
    # Divide into connected regions
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
            
        # Start a new region
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        
        # BFS to find connected points
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            
            # Check 4 adjacent directions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                
                # Check if this adjacent point is in the topk list
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        
        regions.append(region)
    
    # Calculate the average activation value for each region
    region_scores = []
    region_centers = []
    region_points = []
    
    for region in regions:
        # region score = its max activation
        avg_score = max(item[3] for item in region)

        region_scores.append(avg_score)

        # Calculate normalized center coordinates for each patch, then take the average
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()

        for y, x, _, score in region:
            # Normalized coordinates of the center point for each patch
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

            y_coords.add(center_y)
            x_coords.add(center_x)

        region_points.append(normalized_centers)

        # Calculate the average of normalized coordinates as the region center
        if not rect_center:
            # Weighted average
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            avg_center_x, avg_center_y = weighted_x, weighted_y
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))
        
    # Select the region with the highest average activation value
    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        # Outputs:
        # 1. best_point: the center point of the region with the highest average activation value
        # 2. sorted_centers: the center points of all regions, sorted by the average activation value in descending order
        # 3. sorted_scores: the average activation values of all regions, sorted in descending order
        # 4. sorted_points: the normalized center coordinates of all patches, sorted by the average activation value in descending order
        return best_point, sorted_centers, sorted_scores, sorted_points
    else:
        return best_point
