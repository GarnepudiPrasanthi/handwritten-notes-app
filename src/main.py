import os
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import whisper
import cv2
import pytesseract
import textwrap
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Directories - Adjust paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads") # Place uploads outside src
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output") # Place output outside src
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames_tmp") # Place temps outside src
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images_tmp") # Place temps outside src

# Ensure directories exist relative to the project root, not just src
for d in [UPLOAD_FOLDER, OUTPUT_FOLDER, FRAMES_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# Load models globally once
# Note: Model loading might take time and memory. Consider optimizing if needed.
print("Loading Whisper model...")
try:
    # Specify model path if needed, or let whisper find it
    asr_model = whisper.load_model("base")
    print("Whisper model loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    # Handle error appropriately, maybe exit or use a fallback
    asr_model = None # Set to None to avoid errors later if loading failed

print("Loading summarizer...")
try:
    summarizer = pipeline("summarization")
    print("Summarizer loaded.")
except Exception as e:
    print(f"Error loading summarizer pipeline: {e}")
    # Handle error appropriately
    summarizer = None # Set to None

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
    if not asr_model or not summarizer:
         return jsonify({"error": "Models not loaded properly. Check server logs."}), 500

    video = request.files.get("video")
    font = request.files.get("font")

    if not video or not font:
        return jsonify({"error": "Upload both video and font files"}), 400

    # Use safe filenames and ensure directories exist
    # Consider using werkzeug.utils.secure_filename for better security
    video_filename = video.filename
    font_filename = font.filename
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    font_path = os.path.join(UPLOAD_FOLDER, font_filename)
    output_pdf_filename = "handwritten_notes.pdf"
    output_pdf_path = os.path.join(OUTPUT_FOLDER, output_pdf_filename)

    # Ensure target directories exist (redundant if created above, but safe)
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
        # Consider more robust cleanup
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
    # For simplicity, assuming filename is just the base name like "handwritten_notes.pdf"
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

    # 1. Transcribe audio with word timestamps
    print("Starting transcription...")
    if not asr_model:
        raise ValueError("ASR model not loaded.")
    try:
        # Add language detection or specification if needed
        result = asr_model.transcribe(video_path, word_timestamps=True)
        segments = result.get("segments", [])
        print(f"Transcription complete. Found {len(segments)} segments.")
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

    # 2. Extract frames every ~10 seconds for diagram OCR
    print("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Warning: Could not get valid FPS, defaulting to 30.")
        fps = 30
    freq = int(fps * 10) # Extract frame every 10 seconds
    if freq == 0: # Avoid division by zero or infinite loop if fps is very low
        freq = 1
    count, frame_id = 0, 0
    frames_extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Extract frame at the specified frequency
        if count % freq == 0:
            frame_filename = os.path.join(FRAMES_DIR, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
            frames_extracted += 1
        count += 1
    cap.release()
    print(f"Frame extraction complete. {frames_extracted} frames saved.")


    # 3. OCR diagrams from frames
    print("Performing OCR on frames...")
    diagram_texts = []
    # Check if pytesseract is installed and configured
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in PATH.")
        # Depending on requirements, either raise error or skip OCR
        raise EnvironmentError("Tesseract not found. Please install it.")

    ocr_count = 0
    for img_file in sorted(os.listdir(FRAMES_DIR)):
        img_path = os.path.join(FRAMES_DIR, img_file)
        try:
            img = Image.open(img_path)
            # Consider adding image preprocessing here (grayscale, thresholding) if OCR quality is low
            text = pytesseract.image_to_string(img)
            text = text.strip()
            # Filter based on text length or content if needed
            if len(text) > 30: # Keep threshold simple for now
                diagram_texts.append((img_path, text))
                ocr_count += 1
        except Exception as e:
            print(f"Error processing OCR for {img_file}: {e}")
            # Decide whether to skip the frame or raise an error
            continue # Skip this frame
    print(f"OCR complete. {ocr_count} potential diagrams identified.")


    # 4. Summarize transcript segments with 5 min window
    print("Summarizing transcript...")
    if not summarizer:
        raise ValueError("Summarizer model not loaded.")
    summarized_notes = []
    temp_text = ""
    current_time_window_start = 0
    window_duration = 300 # 5 minutes

    # Group segments by time window
    segments_in_window = []
    if segments: # Check if there are any segments
        current_time_window_start = segments[0]['start']
        for segment in segments:
            # Check if segment belongs to the current window
            if segment['start'] < current_time_window_start + window_duration:
                segments_in_window.append(segment['text'])
            else:
                # Process the completed window
                window_text = " ".join(segments_in_window).strip()
                if window_text:
                    try:
                        # Adjust max_length based on input length if needed
                        input_length = len(window_text.split())
                        # Ensure min_length < max_length
                        summary_min_length = min(30, input_length // 3)
                        summary_max_length = max(summary_min_length + 10, min(150, input_length // 2))

                        print(f"Summarizing window starting at {current_time_window_start:.2f}s, length {input_length} words, max_len={summary_max_length}, min_len={summary_min_length}")
                        summary_result = summarizer(window_text, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)
                        if summary_result and isinstance(summary_result, list):
                             summary = summary_result[0]["summary_text"]
                             summarized_notes.append((current_time_window_start, summary))
                             print(f"  -> Summary: {summary[:100]}...")
                        else:
                            print(f"Warning: Unexpected summarizer output for window starting at {current_time_window_start}")

                    except Exception as e:
                        print(f"Error summarizing text window starting at {current_time_window_start}: {e}")
                        # Decide how to handle summarization errors (e.g., skip, use original text)
                        summarized_notes.append((current_time_window_start, "[Summarization Error] " + window_text[:200] + "...")) # Add placeholder

                # Start new window
                current_time_window_start = segment['start']
                segments_in_window = [segment['text']]

        # Process the last window
        window_text = " ".join(segments_in_window).strip()
        if window_text:
            try:
                input_length = len(window_text.split())
                summary_min_length = min(30, input_length // 3)
                summary_max_length = max(summary_min_length + 10, min(150, input_length // 2))
                print(f"Summarizing last window starting at {current_time_window_start:.2f}s, length {input_length} words, max_len={summary_max_length}, min_len={summary_min_length}")
                summary_result = summarizer(window_text, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)
                if summary_result and isinstance(summary_result, list):
                     summary = summary_result[0]["summary_text"]
                     summarized_notes.append((current_time_window_start, summary))
                     print(f"  -> Summary: {summary[:100]}...")
                else:
                     print(f"Warning: Unexpected summarizer output for the last window starting at {current_time_window_start}")
            except Exception as e:
                print(f"Error summarizing the last text window: {e}")
                summarized_notes.append((current_time_window_start, "[Summarization Error] " + window_text[:200] + "..."))
    else:
        print("No transcript segments found to summarize.")

    print(f"Summarization complete. {len(summarized_notes)} notes generated.")


    # 5. Render handwritten images from summaries
    print("Rendering handwritten notes...")
    FONT_SIZE = 28
    PAGE_WIDTH, PAGE_HEIGHT = 1240, 1754 # A4 size in pixels at ~150 DPI
    MARGIN = 100
    LINE_SPACING = 10 # Additional space between lines
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except IOError:
        raise IOError(f"Cannot open font file: {font_path}. Ensure it's a valid .ttf or .otf file.")
    except Exception as e:
        raise RuntimeError(f"Error loading font {font_path}: {e}")

    page_id = 1
    rendered_note_pages = []
    for t, note in summarized_notes:
        # Wrap text to fit page width
        # Estimate average character width (this is approximate)
        avg_char_width = FONT_SIZE * 0.6
        if avg_char_width == 0: avg_char_width = 10 # Avoid division by zero
        max_chars_per_line = (PAGE_WIDTH - 2 * MARGIN) // avg_char_width
        if max_chars_per_line <= 0: max_chars_per_line = 50 # Fallback
        wrapped = textwrap.wrap(note, width=int(max_chars_per_line))

        # Calculate lines per page based on height
        content_height_per_line = FONT_SIZE + LINE_SPACING
        if content_height_per_line == 0: content_height_per_line = 38 # Avoid division by zero
        header_height = FONT_SIZE + 20
        available_height = PAGE_HEIGHT - 2 * MARGIN - header_height
        lines_per_page = available_height // content_height_per_line
        if lines_per_page <= 0: lines_per_page = 20 # Fallback

        # Paginate the wrapped lines
        pages = [wrapped[i : i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)]

        for page_num, lines in enumerate(pages):
            img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
            draw = ImageDraw.Draw(img)
            y = MARGIN

            # Add header (timestamp and page number if multi-page note)
            header = f"🕒 [{int(t//60):02d}:{int(t%60):02d}]"
            if len(pages) > 1:
                header += f" (Page {page_num + 1}/{len(pages)})"
            draw.text((MARGIN, y), header, fill="black", font=font)
            y += header_height # Space after header

            # Draw the lines of the note
            for line in lines:
                draw.text((MARGIN, y), line, fill="black", font=font)
                y += content_height_per_line # Move to next line position

            # Save the page image
            page_filename = os.path.join(IMAGES_DIR, f"page_{page_id:04d}_notes.png")
            img.save(page_filename)
            rendered_note_pages.append(page_filename)
            page_id += 1
    print(f"Handwritten notes rendering complete. {len(rendered_note_pages)} pages created.")


    # 6. Add diagram pages
    print("Rendering diagram pages...")
    rendered_diagram_pages = []
    for path, text in diagram_texts:
        img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(img)

        # Paste the diagram (resized) onto the top half
        try:
            diagram = Image.open(path)
            # Calculate aspect ratio to fit width while maintaining proportions
            img_w, img_h = diagram.size
            target_w = PAGE_WIDTH - 2 * MARGIN
            target_h = (PAGE_HEIGHT // 2) - MARGIN - 10 # Top half minus margins/spacing
            if img_w <= 0 or img_h <= 0: # Avoid division by zero if image dimensions are invalid
                 raise ValueError(f"Invalid image dimensions for {path}: {img_w}x{img_h}")
            ratio = min(target_w / img_w, target_h / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            if new_w <= 0 or new_h <= 0: # Check resized dimensions
                raise ValueError(f"Resized dimensions are invalid: {new_w}x{new_h}")

            diagram = diagram.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center the diagram horizontally
            paste_x = MARGIN + (target_w - new_w) // 2
            paste_y = MARGIN
            img.paste(diagram, (paste_x, paste_y))
        except Exception as e:
            print(f"Error processing diagram image {path}: {e}")
            draw.text((MARGIN, MARGIN), "[Error loading diagram]", fill="red", font=font) # Placeholder
            target_h = PAGE_HEIGHT // 2 # Use default height if diagram fails

        # Add OCR text below the diagram
        y = MARGIN + target_h + 20 # Start text below diagram area
        avg_char_width_diag = FONT_SIZE * 0.6
        if avg_char_width_diag == 0: avg_char_width_diag = 10
        max_chars_per_line_diag = (PAGE_WIDTH - 2 * MARGIN) // avg_char_width_diag
        if max_chars_per_line_diag <= 0: max_chars_per_line_diag = 50

        content_height_per_line_diag = FONT_SIZE + LINE_SPACING
        if content_height_per_line_diag == 0: content_height_per_line_diag = 38
        lines_available = (PAGE_HEIGHT - y - MARGIN) // content_height_per_line_diag
        if lines_available <= 0: lines_available = 10

        wrapped_text = textwrap.wrap(text, width=int(max_chars_per_line_diag))

        for i, line in enumerate(wrapped_text):
            if i >= lines_available:
                draw.text((MARGIN, y), "...", fill="black", font=font) # Indicate truncated text
                break
            draw.text((MARGIN, y), line, fill="black", font=font)
            y += content_height_per_line_diag

        # Save the diagram page
        page_filename = os.path.join(IMAGES_DIR, f"page_{page_id:04d}_diagram.png")
        img.save(page_filename)
        rendered_diagram_pages.append(page_filename)
        page_id += 1
    print(f"Diagram pages rendering complete. {len(rendered_diagram_pages)} pages created.")


    # 7. Combine all images into PDF
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
            try: error_font = ImageFont.load_default() # Basic fallback
            except: error_font = None # No font available

        if error_font:
            draw.text((MARGIN, MARGIN), "Error: No content generated for PDF.", fill="red", font=error_font)
        else:
            draw.text((MARGIN, MARGIN), "Error: No content generated for PDF.", fill="red")
        img.save(output_pdf_path)
        print("PDF generation complete (created error PDF).")
        return # Exit function early

    images_to_save = []
    try:
        # Open first image
        img1 = Image.open(image_files[0])
        img1 = img1.convert('RGB') # Ensure RGB for saving as PDF

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
        raise # Re-raise the exception to be caught by the caller


# Run the app
# Use Gunicorn or Waitress for production deployment instead of Flask's built-in server
if __name__ == "__main__":
    # This block is mainly for local development testing if needed.
    # Render will use a command like `gunicorn src.main:app` specified elsewhere.
    print("Running Flask development server (for testing only)...")
    # Bind to 0.0.0.0 to be accessible within the sandbox network if needed for testing
    # Use a port like 8080 or similar
    # app.run(debug=False, host='0.0.0.0', port=8080)
    pass # Keep this empty for Render deployment

