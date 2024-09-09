FROM python:3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code into the container
COPY . .

# Expose the port your application runs on
EXPOSE 8000

# Number of workers
ENV WORKERS=1

# Start the application
CMD ["fastapi", "run", "--workers", "${WORKERS}", "main.py"]
