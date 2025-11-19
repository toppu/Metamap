# Podman/Docker Setup Guide

Quick reference for running the Microbiome Analysis application with Podman or Docker.

## Installation

**macOS:**
```bash
brew install podman podman-compose
podman machine init --cpus 2 --memory 4096
podman machine start
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install -y podman podman-compose

# Fedora/RHEL
sudo dnf install -y podman podman-compose
```

## Quick Start

```bash
# Start the application
podman-compose up -d --build

# View logs
podman-compose logs -f

# Stop the application
podman-compose down
```

Access at: **http://localhost:8080**

## Common Commands

```bash
# Rebuild after code changes
podman-compose up --build -d

# Check status
podman ps

# View logs
podman-compose logs -f

# Shell access
podman exec -it mcb-microbiome-app bash

# Clean up
podman-compose down
podman system prune -a --volumes
```

## Development Mode

Enable live code reloading in `compose.yml`:
1. Uncomment volume mounts for your code files
2. Set `STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto`
3. Restart: `podman-compose up -d`

## Troubleshooting

**Port already in use:**
```bash
lsof -i :8080 | grep LISTEN
kill -9 <PID>
```

**Container won't start:**
```bash
podman-compose logs
```

**Out of memory:**
```bash
# macOS: increase Podman machine memory
podman machine stop
podman machine set --memory 8192
podman machine start
```

**Permission errors:**
```bash
chmod -R 755 data/
```

## Using Docker Instead

All commands work with Docker by replacing `podman` with `docker`:

```bash
docker-compose up -d --build
docker-compose logs -f
docker-compose down
```

## Additional Resources

- [Podman Documentation](https://docs.podman.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
