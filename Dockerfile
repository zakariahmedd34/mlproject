# -----------------------
# 1. Base image
# -----------------------
FROM python:3.8-slim
# -----------------------
# 2. Set the working directory
# -----------------------
WORKDIR /app
# -----------------------
# 3. Copy requirements first (for caching)
# -----------------------
COPY requirements.txt /app/
# -----------------------
# 4. Install Python dependencies
# -----------------------
RUN pip install --upgrade pip

RUN pip install --no-cache-dir --default-timeout=2000 \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt
# -----------------------
# 5. Copy the rest of the project
# -----------------------
COPY . .
# -----------------------
# 6. Expose port for Flask
# -----------------------
EXPOSE 5000
# -----------------------
# 7. Command to run the app
# -----------------------
CMD ["python", "app.py"]