# Stage 1: Build stage
FROM python:3.8 AS builder

WORKDIR /build

# Copy only necessary files
COPY app/ app/
COPY params.py .
COPY predict.py .
COPY config/predict.yml config/predict.yml
COPY transforms/transform.py transforms/transform.py

# Stage 2: Final stage
FROM python:3.8
LABEL maintainer="Liam Barstad"

#WORKDIR /opt

# Copy files from the builder stage
COPY --from=builder build/ .

# Install production dependencies
RUN pip install --no-cache-dir -r app/requirements.txt
RUN ls -la

# Set the entry point and expose the port
EXPOSE 8080
CMD ["python", "app/app.py"]
