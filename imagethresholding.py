import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog, messagebox, Scale, Button, Label, Frame
import os

def open_image():
    """Open an image file using a file dialog."""
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"),
            ("All Files", "*.*")
        )
    )
    return file_path

def save_image(image):
    """Save the processed image to a file."""
    file_path = filedialog.asksaveasfilename(
        title="Save Processed Image",
        defaultextension=".png",
        filetypes=(
            ("PNG Files", "*.png"),
            ("JPEG Files", "*.jpg;*.jpeg"),
            ("BMP Files", "*.bmp"),
            ("TIFF Files", "*.tiff"),
            ("All Files", "*.*")
        )
    )
    if file_path:
        cv2.imwrite(file_path, image)
        messagebox.showinfo("Success", f"Image saved at {file_path}")

def remove_shadow(image):
    """Remove shadows while preserving text clarity."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Additional contrast enhancement
    lab = cv2.cvtColor(final, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Adjust the L channel
    l = cv2.add(l, 10)
    
    # Merge channels
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

class ImageProcessor:
    def __init__(self):
        self.root = Tk()
        self.root.title("Image Processing Controls")
        self.image = None
        self.processed_image = None
        self.setup_gui()

    def setup_gui(self):
        # Create frame for controls
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # Threshold value slider
        Label(control_frame, text="Block Size (odd number):").pack()
        self.block_size = Scale(control_frame, from_=3, to=31, resolution=2, orient='horizontal')
        self.block_size.set(15)
        self.block_size.pack()

        Label(control_frame, text="C Value:").pack()
        self.c_value = Scale(control_frame, from_=0, to=20, resolution=1, orient='horizontal')
        self.c_value.set(8)
        self.c_value.pack()

        # Buttons
        Button(control_frame, text="Open Image", command=self.open_and_process).pack(pady=5)
        Button(control_frame, text="Process", command=self.update_image).pack(pady=5)
        Button(control_frame, text="Save", command=lambda: save_image(self.processed_image)).pack(pady=5)

    def open_and_process(self):
        input_path = open_image()
        if input_path:
            self.image = cv2.imread(input_path)
            if self.image is None:
                messagebox.showerror("Error", "Could not read the image!")
                return
            self.update_image()

    def process_image(self, image, block_size, c_value):
        """Process the input image with given parameters while retaining Code 2's sharpness."""
        # Step 1: Remove shadows while preserving text
        no_shadow = remove_shadow(image)
        
        # Step 2: Convert the shadow-free image to grayscale
        gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Apply bilateral filtering to preserve edges while smoothing
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Step 4: Adaptive thresholding to create a binary mask
        mask = cv2.adaptiveThreshold(
            smooth, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverted binary mask for better sharpness
            block_size,
            c_value
        )
        
        # Step 5: Convert the binary mask to 3-channel for blending
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        inverted_mask = cv2.bitwise_not(mask_3d)  # Invert for background handling
        
        # Step 6: Create a white background
        white_background = np.full_like(no_shadow, 255)
        
        # Step 7: Extract colored text using the original image and mask
        colored_text = cv2.bitwise_and(image, mask_3d)  # Retain original text color
        
        # Step 8: Apply white background where the mask indicates no text
        background_part = cv2.bitwise_and(white_background, inverted_mask)
        
        # Step 9: Combine colored text and white background
        processed = cv2.add(colored_text, background_part)
        
        return processed


    def update_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        block_size = self.block_size.get()
        c_value = self.c_value.get()
        
        self.processed_image = self.process_image(self.image, block_size, c_value)
        
        # Display the input and output images
        cv2.imshow("Input Image", self.image)
        cv2.imshow("Processed Image", self.processed_image)

    def run(self):
        self.root.mainloop()

def main():
    app = ImageProcessor()
    app.run()

if __name__ == "__main__":
    main()