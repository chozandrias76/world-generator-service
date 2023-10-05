import matplotlib.pyplot as plt
import numpy as np
import cv2


class GeoTiffRotator:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.color_img = cv2.imread(input_file_path)
        self.largest_contour = None
        self.largest_angle = None
        self.converted_img = None

    def display(self):
        if self.converted_img is None:
            plt.imshow(cv2.cvtColor(self.color_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(self.converted_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    def run(self):
        self.deskew()
        self.crop_square(padding_percent=-0.001)

    def deskew(self):
        self.compute_skew()

        if self.largest_angle is None:
            return None
        # load in grayscale:
        img = cv2.imread(self.input_file_path, 0)

        # invert the colors of our image:
        cv2.bitwise_not(img, img)

        # compute the minimum bounding box:
        non_zero_pixels = cv2.findNonZero(img)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)

        root_mat = cv2.getRotationMatrix2D(center, self.largest_angle, 1)
        rows, cols = img.shape
        rotated = cv2.warpAffine(
            self.color_img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC
        )

        # Border removing:
        sizex = np.int0(wh[0])
        sizey = np.int0(wh[1])
        if theta > -45:
            temp = sizex
            sizex = sizey
            sizey = temp
        self.converted_img = cv2.getRectSubPix(rotated, (sizey, sizex), center) # type: ignore

    def compute_skew(self):
        image = cv2.imread(self.input_file_path)
        line_thickness = int(image.shape[1] * 0.005)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to binary using a value of 1 as the threshold
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area in descending order and keep the largest one
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # Initialize list to store angles
        angles = []

        # Calculate line thickness as a fraction of image width
        line_thickness = int(image.shape[1] * 0.005)

        # Draw rectangle for the largest contour
        for contour in sorted_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(
                image, [box], 0, (0, 0, 255), line_thickness  # type: ignore
            )

            # Angle
            angle = rect[-1]
            if angle < -45:
                angle += 90
            angles.append(angle)

        # Since we are keeping only one contour, the angle will be that of the largest square-like shape
        largest_contour = sorted_contours[0][0][0][0] if sorted_contours else None
        largest_angle = angles[0] if angles else None
        self.largest_contour = largest_contour
        self.largest_angle = largest_angle

        return image

    def crop_square(self, padding_percent: float = 0):
        if self.converted_img is None:
            return None
        # Convert to grayscale for simplicity
        gray_image = np.mean(self.converted_img[:, :, :3], axis=2).astype(np.uint8)

        # Find non-black pixels
        non_zero_pixels = np.argwhere(gray_image > 0)

        # Get the coordinates for cropping
        y_min, x_min = np.min(non_zero_pixels, axis=0)
        y_max, x_max = np.max(non_zero_pixels, axis=0)

        # Calculate padding
        y_padding = int((y_max - y_min) * padding_percent)
        x_padding = int((x_max - x_min) * padding_percent)

        print("y_padding", y_padding)
        print("x_padding", x_padding)

        # Apply padding (ensuring boundaries)
        y_min = max(y_min - y_padding, 0)
        x_min = max(x_min - x_padding, 0)
        y_max = min(y_max + y_padding, self.converted_img.shape[0])
        x_max = min(x_max + x_padding, self.converted_img.shape[1])
        # Crop the image
        self.converted_img = self.converted_img[y_min:y_max, x_min:x_max]
