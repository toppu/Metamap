# Combined Python + R service for production deployment
# Using Bioconductor as base to ensure R packages build correctly from source
FROM bioconductor/bioconductor_docker:RELEASE_3_22

WORKDIR /app

# ============================================================================
# STEP 1: Install system dependencies (build tools, utilities)
# ============================================================================
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y --no-install-recommends \
    # Build tools needed for Python packages from source
    build-essential \
    gfortran \
    # System libraries for matplotlib
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    # Network utilities
    curl \
    ca-certificates \
    netcat-openbsd \
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# STEP 2: Install R packages (Bioconductor, ANCOMBC, etc.)
# ============================================================================
COPY install_packages_bioc.R /tmp/install_packages_bioc.R
RUN Rscript /tmp/install_packages_bioc.R && rm /tmp/install_packages_bioc.R

# Install Rserve for R service communication
RUN R -e "install.packages('Rserve', repos='http://www.rforge.net/', Ncpus=4)"

# Create Rserve configuration
RUN mkdir -p /etc/Rserve && \
    echo "host 127.0.0.1" > /etc/Rserve/Rserv.conf && \
    echo "port 6311" >> /etc/Rserve/Rserv.conf && \
    echo "encoding utf8" >> /etc/Rserve/Rserv.conf && \
    echo "remote enable" >> /etc/Rserve/Rserv.conf && \
    echo "auth disable" >> /etc/Rserve/Rserv.conf && \
    cat /etc/Rserve/Rserv.conf

# Copy R support files
RUN mkdir -p /app/src/utils
COPY src/utils/r_support.R /app/src/utils/r_support.R

# ============================================================================
# STEP 3: Install Python packages
# ============================================================================
# Create a virtual environment to bypass EXTERNALLY-MANAGED restrictions
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install a newer setuptools version that's compatible with Python 3.12
RUN pip install --no-cache-dir --upgrade pip 'setuptools>=68.0.0' wheel

# Copy and install Python packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# STEP 4: Copy application files
# ============================================================================
COPY app.py /app/
COPY src/ /app/src/
COPY data/ /app/data/
COPY assets/ /app/assets/
COPY .streamlit/ /app/.streamlit/

# Ensure correct permissions for data directory
RUN mkdir -p /app/ims && chmod -R 755 /app/data /app/ims

# ============================================================================
# STEP 5: Create startup script (R service + Python app)
# ============================================================================
RUN echo '#!/bin/bash' > /usr/local/bin/start-services.sh && \
    echo 'set -e' >> /usr/local/bin/start-services.sh && \
    echo '' >> /usr/local/bin/start-services.sh && \
    echo '# Start Rserve in background' >> /usr/local/bin/start-services.sh && \
    echo 'echo "=== Starting Rserve on 127.0.0.1:6311 ==="' >> /usr/local/bin/start-services.sh && \
    echo 'R CMD Rserve --no-save --RS-conf /etc/Rserve/Rserv.conf' >> /usr/local/bin/start-services.sh && \
    echo 'sleep 3' >> /usr/local/bin/start-services.sh && \
    echo '' >> /usr/local/bin/start-services.sh && \
    echo '# Verify Rserve is running' >> /usr/local/bin/start-services.sh && \
    echo 'if nc -z 127.0.0.1 6311 2>/dev/null; then' >> /usr/local/bin/start-services.sh && \
    echo '  echo "✅ Rserve is listening on 127.0.0.1:6311"' >> /usr/local/bin/start-services.sh && \
    echo 'else' >> /usr/local/bin/start-services.sh && \
    echo '  echo "❌ Rserve failed to start"' >> /usr/local/bin/start-services.sh && \
    echo '  exit 1' >> /usr/local/bin/start-services.sh && \
    echo 'fi' >> /usr/local/bin/start-services.sh && \
    echo '' >> /usr/local/bin/start-services.sh && \
    echo '# Start Streamlit application' >> /usr/local/bin/start-services.sh && \
    echo 'echo "=== Starting Streamlit on 0.0.0.0:8080 ==="' >> /usr/local/bin/start-services.sh && \
    echo 'exec streamlit run --server.port=8080 --server.address=0.0.0.0 app.py' >> /usr/local/bin/start-services.sh && \
    chmod +x /usr/local/bin/start-services.sh

# ============================================================================
# STEP 6: Configure container runtime
# ============================================================================
# Expose Streamlit port (Rserve on 127.0.0.1 only, not exposed)
EXPOSE 8080

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false \
    STREAMLIT_BROWSER_SERVER_ADDRESS=metamap-toolbox.azurewebsites.net \
    STREAMLIT_BROWSER_SERVER_PORT=443 \
    R_SERVICE_URL=http://127.0.0.1:6311 \
    PYTHONWARNINGS="ignore::DeprecationWarning:pkg_resources"

# Start both services
CMD ["/usr/local/bin/start-services.sh"]
