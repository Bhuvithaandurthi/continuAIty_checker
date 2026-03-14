# Use a lightweight Python image
FROM python:3.9-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# 4. Install Torch CPU first (Separate step to prevent timeouts)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Copy and install the rest of requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the code
COPY . .

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Command to run (Ensure your file is named app.py)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]