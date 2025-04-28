import easyocr
import cv2
import numpy as np
import pandas as pd
from google.colab import files
import matplotlib.pyplot as plt

# Initialize EasyOCR Reader
def initialize_reader():
    print("Initializing OCR...")
    return easyocr.Reader(['en'], gpu=True)

# Display Image
def show_image(image, title="Image"):
    plt.figure(figsize=(15, 10))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Preprocess Image for Table Extraction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Denoise with morphological closing
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned, image

# Detect Table Rows and Columns
def detect_table_structure(thresh_image):
    # Horizontal and vertical kernel for lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Detect lines
    horizontal_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine the two lines
    table_lines = cv2.add(horizontal_lines, vertical_lines)
    return table_lines

# Extract Cells and Read Text
def extract_cells_and_text(original_image, table_lines, reader):
    # Find table cell contours
    contours, _ = cv2.findContours(table_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours top-to-bottom, left-to-right
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

    row_cells = {}
    current_row_y = -1
    row_index = 0

    # Extract cell text
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)

        # Skip small cells
        if w < 30 or h < 20:
            continue

        # New row detection
        if current_row_y == -1 or abs(y - current_row_y) > 10:
            row_index += 1
            row_cells[row_index] = []
            current_row_y = y

        # Crop cell image
        cell_img = original_image[y:y+h, x:x+w]
        text = reader.readtext(cell_img, detail=0)
        cell_text = ' '.join(text).strip() if text else ""

        # Append to row
        row_cells[row_index].append(cell_text)

    return row_cells

# Convert Table Data to Excel
def save_to_excel(table_data):
    # Define correct column headers
    headers = ['Sl. No', 'USN', 'Name']

    # Clean and prepare table rows
    data_rows = [row for row in table_data.values() if len(row) == len(headers)]

    # Convert to DataFrame
    df = pd.DataFrame(data_rows, columns=headers)

    # Save as Excel
    output_file = 'attendance_output.xlsx'
    df.to_excel(output_file, index=False)
    print(f"File saved: {output_file}")
    files.download(output_file)

# Main Function
def main():
    print("Upload the attendance image...")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]

    try:
        # Initialize EasyOCR
        reader = initialize_reader()

        # Preprocess Image
        print("Preprocessing the image...")
        preprocessed_image, original_image = preprocess_image(filename)
        show_image(preprocessed_image, "Preprocessed Image")

        # Detect Table Lines
        print("Detecting table structure...")
        table_structure = detect_table_structure(preprocessed_image)
        show_image(table_structure, "Detected Table Structure")

        # Extract Cells and Text
        print("Extracting table data...")
        table_data = extract_cells_and_text(original_image, table_structure, reader)

        # Convert and Save Data to Excel
        print("Saving to Excel...")
        save_to_excel(table_data)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the Script
if __name__ == "__main__":
    main()
