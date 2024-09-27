# Use a Python 3.10 image based on Debian Bookworm as your base image
FROM python:3.10-bookworm

# Update the package list and install GDAL's dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin

# Set environment variables to ensure that the Python GDAL bindings
# can find and use the GDAL installation
ENV GDAL_CONFIG=/usr/bin/gdal-config
# ENV GDAL_VERSION=$(gdal-config --version)

# Your existing Dockerfile instructions follow...