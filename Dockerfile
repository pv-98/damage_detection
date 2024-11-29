# Base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000


CMD ["python", "main.py", "--serve" ,"--host", "0.0.0.0", "--port", "8000"]
