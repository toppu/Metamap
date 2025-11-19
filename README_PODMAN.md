# Running Microbiome Analysis App with Podman Compose

This guide explains how to run the Microbiome Analysis application using Podman Compose.

## Prerequisites

### Install Podman and Podman Compose

**macOS:**
```bash
brew install podman podman-compose
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y podman podman-compose
```

**Fedora/RHEL:**
```bash
sudo dnf install -y podman podman-compose
```

### Initialize Podman Machine (macOS/Windows only)

On macOS, you need to create and start a Podman machine:
```bash
podman machine init --cpus 2 --memory 4096 --disk-size 50
podman machine start
```

Check status:
```bash
podman machine list
```

## Quick Start

### 1. Build and Run

From the project root directory:

```bash
# Build and start the container
podman-compose up --build

# Or run in detached mode (background)
podman-compose up -d --build
```

### 2. Access the Application

Once the container is running and healthy:
- **URL:** http://localhost:8080
- **Startup time:** ~60 seconds (R packages need to load)

### 3. Stop the Application

```bash
# Stop containers
podman-compose down

# Stop and remove volumes (caution: deletes data!)
podman-compose down -v
```

## Common Commands

### View Logs
```bash
# Follow logs in real-time
podman-compose logs -f

# View last 100 lines
podman-compose logs --tail 100
```

### Rebuild After Code Changes
```bash
# Rebuild and restart
podman-compose up --build -d

# Force rebuild (ignore cache)
podman-compose build --no-cache
podman-compose up -d
```

### Check Container Status
```bash
# List running containers
podman ps

# View container details
podman-compose ps

# Check health status
podman inspect mcb-microbiome-app | grep -A 10 Health
```

### Execute Commands Inside Container
```bash
# Open shell in running container
podman exec -it mcb-microbiome-app bash

# Run R script
podman exec -it mcb-microbiome-app Rscript -e "installed.packages()"

# Check Python packages
podman exec -it mcb-microbiome-app pip list
```

### Clean Up Everything
```bash
# Stop and remove containers, networks
podman-compose down

# Remove all unused images, containers, volumes
podman system prune -a --volumes

# Nuclear option: reset everything
podman machine stop
podman machine rm
podman machine init --cpus 2 --memory 4096 --disk-size 50
podman machine start
```

## Development Mode

For active development with live code reloading:

1. Uncomment the volume mounts in `podman-compose.yml`:
```yaml
volumes:
  - ./Data:/app/Data:rw
  - ./app.py:/app/app.py:ro
  - ./helpers.py:/app/helpers.py:ro
  - ./r_support.R:/app/r_support.R:ro
  - ./src/pages/Ecological_Diversity.py:/app/pages/Ecological_Diversity.py:ro
  - ./src/pages/Statistical_Analysis.py:/app/pages/Statistical_Analysis.py:ro
```

2. Enable file watcher in environment variables:
```yaml
environment:
  - STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto  # Change from 'none'
```

3. Restart:
```bash
podman-compose up -d
```

Now code changes will be reflected immediately!

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8080
lsof -i :8080

# Kill it
kill -9 <PID>

# Or change port in podman-compose.yml
ports:
  - "8501:8080"  # Now accessible at http://localhost:8501
```

### Container Won't Start
```bash
# Check logs
podman-compose logs

# Check R installation logs
podman exec -it mcb-microbiome-app cat /app/install.log

# Verify R packages
podman exec -it mcb-microbiome-app Rscript -e "library(vegan); library(ANCOMBC)"
```

### Slow Build Times
The first build takes **15-30 minutes** because:
- R is compiled from source (~10 min)
- Bioconductor packages are installed (~15 min)

Subsequent builds are faster (~2-5 min) thanks to Docker layer caching.

**Speed up builds:**
```bash
# Use more CPUs (if available)
podman machine stop
podman machine rm
podman machine init --cpus 4 --memory 8192
podman machine start
```

### Out of Memory
If build fails with memory errors:

```bash
# Increase machine memory
podman machine stop
podman machine set --memory 8192  # 8GB
podman machine start

# Or adjust resource limits in podman-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

### Permission Errors on Data Directory
```bash
# Fix permissions
chmod -R 755 Data/

# Or run container with specific user
podman-compose run --user $(id -u):$(id -g) mcb-app
```

### Healthcheck Failing
```bash
# Check if curl is available
podman exec -it mcb-microbiome-app curl -f http://localhost:8080/_stcore/health

# Disable healthcheck temporarily (podman-compose.yml)
# Comment out the healthcheck section
```

## Performance Tuning

### Adjust Resource Limits

Edit `podman-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # More CPU for faster analysis
      memory: 8G       # More RAM for large datasets
    reservations:
      cpus: '2.0'
      memory: 4G
```

### Optimize Podman Machine
```bash
# Check current settings
podman machine inspect

# Recreate with more resources
podman machine stop
podman machine rm
podman machine init \
  --cpus 4 \
  --memory 8192 \
  --disk-size 100
podman machine start
```

## Production Deployment

### Using Podman on a Server

```bash
# Clone repository
git clone <repo-url>
cd Mcb_Website

# Build and run
podman-compose up -d --build

# Set up systemd service for auto-start
podman generate systemd --new --name mcb-microbiome-app > /etc/systemd/system/mcb-app.service
systemctl enable mcb-app.service
systemctl start mcb-app.service
```

### Behind a Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name microbiome.yourdomain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_read_timeout 86400;
    }
}
```

### Environment-Specific Configuration

Create `.env` file:
```bash
# .env
STREAMLIT_SERVER_PORT=8080
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

Update `podman-compose.yml`:
```yaml
services:
  mcb-app:
    env_file:
      - .env
```

## Backup and Restore Data

### Backup
```bash
# Backup Data directory
tar -czf mcb_data_backup_$(date +%Y%m%d).tar.gz Data/

# Backup with Podman volumes
podman volume export mcb_data > mcb_data_backup.tar
```

### Restore
```bash
# Restore Data directory
tar -xzf mcb_data_backup_20251106.tar.gz

# Restore volume
podman volume import mcb_data mcb_data_backup.tar
```

## Differences: Podman vs Docker

Podman is designed to be a drop-in replacement for Docker with these differences:

| Feature | Podman | Docker |
|---------|--------|--------|
| **Daemon** | Daemonless | Requires Docker daemon |
| **Root** | Rootless by default | Requires root or group membership |
| **Systemd** | Native integration | Requires additional setup |
| **Security** | More secure (no daemon) | Less secure (daemon runs as root) |
| **Commands** | `podman` instead of `docker` | `docker` |
| **Compose** | `podman-compose` (separate tool) | `docker compose` (native) |

All Docker commands work with Podman:
```bash
# Create alias if you're used to Docker
alias docker=podman
alias docker-compose=podman-compose
```

## Monitoring

### View Resource Usage
```bash
# Real-time stats
podman stats mcb-microbiome-app

# One-time check
podman stats --no-stream
```

### Log Monitoring
```bash
# Save logs to file
podman-compose logs > app_logs_$(date +%Y%m%d).log

# Monitor errors only
podman-compose logs | grep -i error

# Monitor R package loading
podman-compose logs | grep -i "loading\|library"
```

## CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/build.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Podman
        run: |
          sudo apt-get update
          sudo apt-get install -y podman
      
      - name: Build image
        run: podman build -t mcb-app .
      
      - name: Test container
        run: |
          podman run -d --name test-app -p 8080:8080 mcb-app
          sleep 60
          curl -f http://localhost:8080/_stcore/health
```

## Additional Resources

- [Podman Documentation](https://docs.podman.io/)
- [Podman Compose GitHub](https://github.com/containers/podman-compose)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [R Package Installation Guide](https://cran.r-project.org/)

## Support

If you encounter issues:
1. Check logs: `podman-compose logs -f`
2. Verify R packages: `podman exec -it mcb-microbiome-app Rscript -e "installed.packages()"`
3. Check health: `podman inspect mcb-microbiome-app`
4. Open an issue on GitHub with full logs

---

**Last Updated:** November 6, 2025
