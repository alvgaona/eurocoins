import cv2
import numpy as np
import os
import glob

if __name__ == "__main__":
    # Get all image files from the train directory
    train_dir = "resources/custom-euro-coins"
    image_files = glob.glob(os.path.join(train_dir, "*.jpg"))
    image_files.sort()  # Sort for consistent order

    print(f"Found {len(image_files)} images in train directory")

    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_files)}] Processing: {filename}")

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load image from {image_path}")
            continue

        print(f"Image loaded successfully. Shape: {image.shape}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise using bilateral filter to preserve edges while reducing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Detect circles using Circular Hough Transform
        circles = cv2.HoughCircles(
            denoised,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=gray.shape[0] / 16,  # Increased to avoid overlapping detections
            param1=50,  # Higher threshold for internal Canny
            param2=50,   # Lower accumulator threshold for detection
            minRadius=10,  # Adjusted for typical coin sizes
            maxRadius=300
        )

        # Create a copy of the original image for drawing
        result_image = image.copy()
        circle_count = 0

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circle_count = len(circles)
            print(f"Found {circle_count} circles")

            # Draw the circles
            for (x, y, r) in circles:
                cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)  # Green circle
                cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)  # Red center

        else:
            print("No circles detected")

        # Create a combined image showing edges and detected circles side by side
        window_title = f"[{i}/{len(image_files)}] {filename} - {circle_count} circles"

        # Create and configure the window
        window_name = "Edges | Detected Circles"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Set window size (width, height)
        window_width = 1200
        window_height = 600
        cv2.resizeWindow(window_name, window_width, window_height)

        # Center the window on screen
        # Get screen dimensions (approximate)
        screen_width = 1512  # Adjust if needed for your screen
        screen_height = 982  # Adjust if needed for your screen
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        cv2.moveWindow(window_name, x_pos, y_pos)

        # Display the combined image
        cv2.imshow(window_name, result_image)
        cv2.setWindowTitle(window_name, window_title)

        # Wait for a key press before moving to next image
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Allow early exit with 'q' key
        if key == ord('q'):
            print("Exiting early...")
            break

    print("\nProcessing complete!")
