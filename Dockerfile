FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and dataset
COPY . .

# Environment configuration
ENV PORT=8080
EXPOSE 8080

CMD ["python", "app/web_app.py"]
