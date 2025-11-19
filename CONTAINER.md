# Container Usage Guide

## Pull and Run the Container

### Using Docker

```bash
# Pull the latest image
docker pull ghcr.io/toppu/metamap:latest

# Run the container
docker run -p 8080:8080 ghcr.io/toppu/metamap:latest

# Run with custom data directory mounted
docker run -p 8080:8080 -v $(pwd)/data:/app/data ghcr.io/toppu/metamap:latest
```

### Using Podman

```bash
# Pull the latest image
podman pull ghcr.io/toppu/metamap:latest

# Run the container
podman run -p 8080:8080 ghcr.io/toppu/metamap:latest

# Run with custom data directory mounted
podman run -p 8080:8080 -v $(pwd)/data:/app/data:Z ghcr.io/toppu/metamap:latest
```

## Access the Application

Once running, open your browser and navigate to:
```
http://localhost:8080
```

## Available Tags

- `latest` - Latest stable release from main branch
- `v*.*.*` - Specific version tags (e.g., `v1.0.0`)
- `main-<sha>` - Specific commit from main branch
- Platform support: `linux/amd64`, `linux/arm64`

## Environment Variables

You can customize the application using environment variables:

```bash
docker run -p 8080:8080 \
  -e STREAMLIT_SERVER_PORT=8080 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  ghcr.io/toppu/metamap:latest
```

## Docker Compose

See `compose.yml` for Docker Compose configuration.

## Building Locally

```bash
# Build the image
docker build -t metamap:local .

# Run locally built image
docker run -p 8080:8080 metamap:local
```

## Automated Builds

This repository uses GitHub Actions to automatically build and publish container images:

- **On push to main**: Creates `latest` tag and commit-specific tag
- **On version tag** (e.g., `v1.0.0`): Creates version tags
- **On pull request**: Builds but doesn't push (validation only)

### Creating a Release

```bash
# Tag a new version
git tag v1.0.0
git push origin v1.0.0

# This will trigger automatic build and push to ghcr.io
```

## Troubleshooting

### Container won't start
Check logs:
```bash
docker logs <container-id>
```

### Permission issues with mounted volumes
Use the `:z` or `:Z` flag for SELinux systems:
```bash
docker run -p 8080:8080 -v $(pwd)/data:/app/data:Z ghcr.io/toppu/metamap:latest
```

### Port already in use
Use a different port:
```bash
docker run -p 8501:8080 ghcr.io/toppu/metamap:latest
# Access at http://localhost:8501
```

## Health Check

The container includes a health check endpoint:
```bash
curl http://localhost:8080/_stcore/health
```

## Support

For issues, please visit: https://github.com/toppu/metamap/issues
