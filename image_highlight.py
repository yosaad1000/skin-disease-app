
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def extract_lbp_features(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Parameters for LBP
    radius = 1
    n_points = 8 * radius
    # Compute LBP
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    # Normalize LBP values to range [0, 1]
    lbp = lbp / lbp.max()
    return lbp

def fill_holes(mask):
    # Invert mask: holes become white
    inverted_mask = cv2.bitwise_not(mask)
    # Flood fill from point (0, 0)
    h, w = inverted_mask.shape[:2]
    flood_filled = inverted_mask.copy()
    mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask_floodfill, (0, 0), 255)
    # Invert flood-filled image
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    # Combine with original mask to fill holes
    filled_mask = mask | flood_filled_inv
    return filled_mask

def segment_and_highlight(image_path, k=2):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    # Extract LBP features
    lbp_features = extract_lbp_features(image)
    
    # Reshape image and LBP features
    height, width = image.shape[:2]
    pixel_values = image.reshape((-1, 3))
    lbp_values = lbp_features.reshape((-1, 1))
    
    # Combine color and texture features
    combined_features = np.hstack((pixel_values, lbp_values))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(combined_features)
    segmented_image = labels.reshape((height, width))
    
    # Highlight the anomalous region
    # Assuming the lesion is in the minority cluster
    unique, counts = np.unique(labels, return_counts=True)
    lesion_cluster = unique[np.argmin(counts)]
    mask = (segmented_image == lesion_cluster).astype(np.uint8) * 255
    
    # Fill holes in the mask
    filled_mask = fill_holes(mask)
    
    # Create an overlay
    overlay = image.copy()
    overlay[filled_mask == 255] = (0, 0, 255)  # Red color for the lesion area
    
    # Blend the original image with the overlay
    alpha = 0.5  # Transparency factor
    highlighted_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return highlighted_image

if __name__ == "__main__":
    try:
        result_image = segment_and_highlight('images melanoma.jpeg')
        # Display the result
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Highlighted Segmented Area without Gaps')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
