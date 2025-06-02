import os
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import cv2
import textwrap

app = Flask(__name__)
CORS(app)

# Directories - Adjust paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames_tmp")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images_tmp")

# Ensure directories exist relative to the project root
for d in [UPLOAD_FOLDER, OUTPUT_FOLDER, FRAMES_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# Serve frontend from the 'frontend' directory at the project root
@app.route("/")
def index():
    # ../frontend relative to src/main.py
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    return send_from_directory(frontend_dir, "index.html")

# Serve other static files from frontend (CSS, JS) if needed
@app.route("/<path:filename>")
def serve_static(filename):
     # ../frontend relative to src/main.py
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    # Basic security check to prevent accessing files outside frontend_dir
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid path"}), 400
    return send_from_directory(frontend_dir, filename)

@app.route("/generate", methods=["POST"])
def generate_notes():
    video = request.files.get("video")
    font = request.files.get("font")

    if not video or not font:
        return jsonify({"error": "Upload both video and font files"}), 400

    # Use safe filenames and ensure directories exist
    video_filename = video.filename
    font_filename = font.filename
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    font_path = os.path.join(UPLOAD_FOLDER, font_filename)
    output_pdf_filename = "handwritten_notes.pdf"
    output_pdf_path = os.path.join(OUTPUT_FOLDER, output_pdf_filename)

    # Ensure target directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        video.save(video_path)
        font.save(font_path)
    except Exception as e:
        print(f"Error saving uploaded files: {e}")
        return jsonify({"error": "Failed to save uploaded files."}), 500

    # Call processing pipeline
    try:
        print(f"Starting note generation for video: {video_filename}, font: {font_filename}")
        generate_handwritten_notes(video_path, font_path, output_pdf_path)
        print("Note generation function completed.")
    except Exception as e:
        print(f"Error during note generation: {e}")
        # Clean up potentially corrupted intermediate files
        shutil.rmtree(FRAMES_DIR, ignore_errors=True)
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)
        # Recreate dirs for next attempt
        os.makedirs(FRAMES_DIR, exist_ok=True)
        os.makedirs(IMAGES_DIR, exist_ok=True)
        return jsonify({"error": f"An error occurred during processing: {e}"}), 500

    # Return the path relative to the server root for download
    print(f"Successfully generated PDF: {output_pdf_path}")
    return jsonify({"pdf_url": f"/download/{output_pdf_filename}"})

@app.route("/download/<filename>")
def download_file(filename):
    # Ensure filename is safe (e.g., prevent directory traversal)
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid filename"}), 400

    safe_path = os.path.join(OUTPUT_FOLDER, filename)
    print(f"Attempting to send file: {safe_path}")

    # Check if file exists and is within the OUTPUT_FOLDER
    if os.path.isfile(safe_path) and os.path.abspath(safe_path).startswith(os.path.abspath(OUTPUT_FOLDER)):
         return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
    else:
        print(f"File not found or invalid path: {safe_path}")
        return jsonify({"error": "File not found"}), 404

def generate_handwritten_notes(video_path, font_path, output_pdf_path):
    # Clean old data and recreate directories
    print("Cleaning up temporary directories...")
    for d in [FRAMES_DIR, IMAGES_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # 1. Extract frames from video (simplified - just extract a few frames)
    print("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Extract 5 frames evenly distributed throughout the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 100  # Default if can't determine frame count
    
    frames_to_extract = min(5, frame_count)
    interval = max(1, frame_count // frames_to_extract)
    
    frames_extracted = 0
    for i in range(frames_to_extract):
        # Set position to extract frame
        position = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames_extracted += 1
    
    cap.release()
    print(f"Frame extraction complete. {frames_extracted} frames saved.")

    # 2. Generate sample notes (instead of transcription and summarization)
    print("Generating sample notes...")
    sample_notes = [
        (0, "Welcome to the class! Today we'll be discussing key concepts in machine learning."),
        (300, "Supervised learning involves training with labeled data. Examples include classification and regression."),
        (600, "Unsupervised learning works with unlabeled data. Clustering and dimensionality reduction are common techniques."),
        (900, "Neural networks are inspired by the human brain and consist of layers of interconnected nodes."),
        (1200, "Deep learning uses multiple layers to progressively extract higher-level features from raw input.")
    ]

    # 3. Render handwritten images from sample notes
    print("Rendering handwritten notes...")
    FONT_SIZE = 28
    PAGE_WIDTH, PAGE_HEIGHT = 1240, 1754  # A4 size in pixels at ~150 DPI
    MARGIN = 100
    LINE_SPACING = 10  # Additional space between lines
    
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception as e:
        raise RuntimeError(f"Error loading font {font_path}: {e}")

    page_id = 1
    rendered_note_pages = []
    
    for t, note in sample_notes:
        # Wrap text to fit page width
        avg_char_width = FONT_SIZE * 0.6
        max_chars_per_line = int((PAGE_WIDTH - 2 * MARGIN) // avg_char_width)
        wrapped = textwrap.wrap(note, width=max_chars_per_line)

        # Calculate lines per page
        content_height_per_line = FONT_SIZE + LINE_SPACING
        header_height = FONT_SIZE + 20
        available_height = PAGE_HEIGHT - 2 * MARGIN - header_height
        lines_per_page = available_height // content_height_per_line

        # Paginate the wrapped lines
        pages = [wrapped[i : i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)]

        for page_num, lines in enumerate(pages):
            img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
            draw = ImageDraw.Draw(img)
            y = MARGIN

            # Add header (timestamp and page number if multi-page note)
            minutes = int(t // 60)
            seconds = int(t % 60)
            header = f"🕒 [{minutes:02d}:{seconds:02d}]"
            if len(pages) > 1:
                header += f" (Page {page_num + 1}/{len(pages)})"
            draw.text((MARGIN, y), header, fill="black", font=font)
            y += header_height  # Space after header

            # Draw the lines of the note
            for line in lines:
                draw.text((MARGIN, y), line, fill="black", font=font)
                y += content_height_per_line  # Move to next line position

            # Save the page image
            page_filename = os.path.join(IMAGES_DIR, f"page_{page_id:04d}_notes.png")
            img.save(page_filename)
            rendered_note_pages.append(page_filename)
            page_id += 1
    
    print(f"Handwritten notes rendering complete. {len(rendered_note_pages)} pages created.")

    # 4. Add diagram pages from extracted frames
    print("Rendering diagram pages...")
    rendered_diagram_pages = []
    
    for i, frame_file in enumerate(sorted(os.listdir(FRAMES_DIR))):
        frame_path = os.path.join(FRAMES_DIR, frame_file)
        
        img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(img)
        
        # Add a title
        title = f"Diagram from video (Frame {i+1})"
        draw.text((MARGIN, MARGIN), title, fill="black", font=font)
        
        # Paste the frame (resized) onto the page
        try:
            diagram = Image.open(frame_path)
            # Calculate aspect ratio to fit width while maintaining proportions
            img_w, img_h = diagram.size
            target_w = PAGE_WIDTH - 2 * MARGIN
            target_h = (PAGE_HEIGHT // 2) - MARGIN - 50  # Top half minus margins/spacing
            
            ratio = min(target_w / img_w, target_h / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            
            diagram = diagram.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center the diagram horizontally
            paste_x = MARGIN + (target_w - new_w) // 2
            paste_y = MARGIN + 40  # Below the title
            img.paste(diagram, (paste_x, paste_y))
            
            # Add a caption
            caption_y = paste_y + new_h + 20
            caption = f"This is a key visual from the video that would normally be processed with OCR."
            draw.text((MARGIN, caption_y), caption, fill="black", font=font)
            
        except Exception as e:
            print(f"Error processing diagram image {frame_path}: {e}")
            draw.text((MARGIN, MARGIN + 50), "[Error loading diagram]", fill="red", font=font)
        
        # Save the diagram page
        page_filename = os.path.join(IMAGES_DIR, f"page_{page_id:04d}_diagram.png")
        img.save(page_filename)
        rendered_diagram_pages.append(page_filename)
        page_id += 1
    
    print(f"Diagram pages rendering complete. {len(rendered_diagram_pages)} pages created.")

    # 5. Combine all images into PDF
    print("Combining images into PDF...")
    # Sort images by filename to maintain order (notes pages first, then diagrams)
    image_files = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])

    if not image_files:
        print("Warning: No images generated to create PDF.")
        # Create a blank PDF with an error message
        img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(img)
        try:
            # Try loading the font again for the error message
            error_font = ImageFont.truetype(font_path, FONT_SIZE)
        except:
            # Fallback to a default font if user font fails here
            try: error_font = ImageFont.load_default()  # Basic fallback
            except: error_font = None  # No font available

        if error_font:
            draw.text((MARGIN, MARGIN), "Error: No content generated for PDF.", fill="red", font=error_font)
        else:
            draw.text((MARGIN, MARGIN), "Error: No content generated for PDF.", fill="red")
        img.save(output_pdf_path)
        print("PDF generation complete (created error PDF).")
        return  # Exit function early

    images_to_save = []
    try:
        # Open first image
        img1 = Image.open(image_files[0])
        img1 = img1.convert('RGB')  # Ensure RGB for saving as PDF

        # Open remaining images
        for f in image_files[1:]:
            img = Image.open(f)
            img = img.convert('RGB')
            images_to_save.append(img)

        # Save as PDF
        print(f"Saving PDF to {output_pdf_path} with {len(image_files)} pages...")
        img1.save(output_pdf_path, save_all=True, append_images=images_to_save)
        print("PDF generation complete.")

    except Exception as e:
        print(f"Error combining images into PDF: {e}")
        raise  # Re-raise the exception to be caught by the caller

# Run the app
if __name__ == "__main__":
    # This block is mainly for local development testing if needed.
    # Render will use a command like `gunicorn src.main:app` specified elsewhere.
    print("Running Flask development server (for testing only)...")
    # Bind to 0.0.0.0 to be accessible within the sandbox network if needed for testing
    # app.run(debug=False, host='0.0.0.0', port=8080)
    pass  # Keep this empty for Render deployment
