# Base image with Python
FROM python:3.9-slim

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the app using uvicorn
CMD ["python", "main.py", "--serve"]