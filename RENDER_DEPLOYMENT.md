# Render Deployment Instructions for Handwritten Notes App

## Prerequisites
1. A [Render](https://render.com) account
2. Git repository with your project code

## Step 1: Prepare Your Repository
1. Create a new Git repository (GitHub, GitLab, etc.)
2. Push all the project files to your repository:
   - `src/main.py` - Flask application backend
   - `frontend/index.html` - Frontend interface
   - `requirements.txt` - Python dependencies
   - `render.yaml` - Render deployment configuration

## Step 2: Connect to Render
1. Log in to your [Render Dashboard](https://dashboard.render.com)
2. Click on "New" and select "Blueprint" from the dropdown menu
3. Connect your Git repository where you pushed the project files
4. Select the repository containing the handwritten notes app

## Step 3: Configure the Blueprint
1. Render will automatically detect the `render.yaml` file
2. Review the configuration settings:
   - Service name: `handwritten-notes-app`
   - Environment: `python`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn src.main:app`
   - Plan: `free` (or choose a paid plan if needed)

## Step 4: Deploy the Application
1. Click "Apply Blueprint" to start the deployment process
2. Render will automatically install dependencies and deploy your application
3. Wait for the build and deployment process to complete (this may take several minutes)

## Step 5: Install Additional Dependencies (if needed)
If the deployment fails due to missing system dependencies, you may need to add them:
1. Go to your service in the Render dashboard
2. Navigate to "Environment" tab
3. Add the following environment variables if needed:
   - Key: `APT_PKGS`, Value: `tesseract-ocr libtesseract-dev ffmpeg`

## Step 6: Access Your Application
1. Once deployment is successful, Render will provide a URL to access your application
2. Click on the URL to open your handwritten notes generator app
3. Test the application by uploading a video and font file

## Troubleshooting
- If you encounter errors, check the logs in the Render dashboard
- Ensure all dependencies are correctly listed in `requirements.txt`
- Verify that the `render.yaml` file is properly configured
- Check if any system dependencies are missing (like Tesseract OCR)

## Important Notes
- The free tier of Render has limited resources, which may affect processing speed
- Large video files may take significant time to process
- The application requires both Whisper and Transformers models, which are downloaded during the first run
