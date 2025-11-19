# GitHub Actions Setup Guide

## What Was Created

1. **`.github/workflows/docker-publish.yml`** - Automated container build and push workflow
2. **`CONTAINER.md`** - Container usage documentation
3. **Updated `README.md`** - Added container registry information

## Next Steps

### 1. Commit and Push Changes

```bash
# Check what was created
git status

# Add all new files
git add .github/workflows/docker-publish.yml CONTAINER.md README.md

# Commit the changes
git commit -m "feat: add GitHub Actions for automated container builds

- Add workflow to build and push Docker images to GHCR
- Support multiple platforms (amd64, arm64)
- Automatic tagging based on git refs
- Add container usage documentation"

# Push to GitHub
git push origin feat/container
```

### 2. Merge to Main

After pushing, you can:
1. Create a Pull Request on GitHub
2. Review and merge to `main`
3. Once merged, the workflow will automatically run and publish your container

### 3. Verify the Build

After merging to main, go to:
- Your GitHub repository
- Click "Actions" tab
- Watch the "Build and Push Docker Image" workflow run

### 4. Access Your Container

Once the workflow completes successfully, your container will be available at:
```
ghcr.io/toppu/metamap:latest
```

### 5. Make Package Public (Important!)

By default, GitHub packages are private. To make it publicly accessible:

1. Go to: https://github.com/toppu?tab=packages
2. Find the `metamap` package
3. Click on it
4. Go to "Package settings" (right sidebar)
5. Scroll down to "Danger Zone"
6. Click "Change visibility" → "Public"

### 6. Create Version Releases (Optional)

To create versioned releases:

```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0
```

This will create additional tags:
- `ghcr.io/toppu/metamap:v1.0.0`
- `ghcr.io/toppu/metamap:1.0`
- `ghcr.io/toppu/metamap:1`
- `ghcr.io/toppu/metamap:latest`

## How the Workflow Works

### Triggers
- **Push to main**: Builds and pushes `latest` + commit SHA tags
- **Version tags** (v*.*.*): Builds and pushes version-specific tags
- **Pull requests**: Builds only (no push, validation)
- **Manual**: Can trigger via GitHub Actions UI

### What It Does
1. Checks out your code
2. Sets up Docker Buildx for multi-platform builds
3. Logs into GitHub Container Registry (using automatic GITHUB_TOKEN)
4. Extracts metadata for tags and labels
5. Builds for both amd64 and arm64
6. Pushes to ghcr.io
7. Creates build attestation for security

### Caching
The workflow uses GitHub Actions cache to speed up subsequent builds:
- First build: ~5-10 minutes
- Subsequent builds: ~2-3 minutes (with cache)

## Troubleshooting

### Build Fails
- Check the Actions logs on GitHub
- Common issues:
  - Dockerfile syntax errors
  - Missing dependencies in requirements.txt
  - Platform-specific build issues

### Can't Pull Container
- Ensure package visibility is set to "Public"
- Check that workflow completed successfully
- Verify package exists at: https://github.com/toppu/metamap/pkgs/container/metamap

### Permission Issues
The workflow uses `GITHUB_TOKEN` which is automatically provided by GitHub Actions. No manual token setup required!

## Alternative: Docker Hub

If you prefer Docker Hub instead:
1. Create account at hub.docker.com
2. Create repository: `toppu/metamap`
3. Add Docker Hub credentials to GitHub Secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
4. Modify workflow to use Docker Hub registry

## Support

For issues, see the GitHub Actions logs or create an issue in your repository.
