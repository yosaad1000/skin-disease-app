import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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

def segment_and_highlight(image_path, k=2):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    # Convert to PIL for better image manipulation
    pil_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image, "RGBA")
    
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
    unique, counts = np.unique(labels, return_counts=True)
    lesion_cluster = unique[np.argmin(counts)]  # Assuming lesion is the smallest cluster
    mask = (segmented_image == lesion_cluster).astype(np.uint8) * 255
    
    # Find contours of the lesion
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours using PIL
    for contour in contours:
        if len(contour) >= 2:  # Ensure the contour has at least two points
            points = [(point[0][0], point[0][1]) for point in contour]
            draw.line(points + [points[0]], fill="black", width=3)  # Outline in black
            draw.polygon(points, outline="black", fill=(255, 0, 0, 100))  # Highlight in red with transparency
    
    return pil_image

if __name__ == "__main__":
    try:
        result_image = segment_and_highlight('images melanoma.jpeg')
        # Display the result
        plt.imshow(result_image)
        plt.title('Outlined and Highlighted Lesion Area')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
