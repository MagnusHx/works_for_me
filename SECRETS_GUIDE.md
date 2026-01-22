# Secrets Management with GCP Secret Manager

## Setup (one-time)

1. **Enable Secret Manager API:**
   ```bash
   gcloud services enable secretmanager.googleapis.com
   ```

2. **Create a secret for wandb:**
   ```bash
   echo -n "your-wandb-api-key" | gcloud secrets create WANDB_API_KEY --data-file=-
   ```

## Option A: Pass secrets to Compute Engine VMs

When you SSH into your VM and run training:

```bash
# Set environment variable in your VM session
export WANDB_API_KEY=$(gcloud secrets versions access latest --secret="WANDB_API_KEY")

# Then run your Docker container with the secret
docker run -e WANDB_API_KEY=$WANDB_API_KEY \
  europe-west1-docker.pkg.dev/proven-cosine-484310-s2/dtumlops/images:train-latest
```

## Option B: Use a startup script for your VM

Create a file `startup-script.sh`:
```bash
#!/bin/bash
export WANDB_API_KEY=$(gcloud secrets versions access latest --secret="WANDB_API_KEY")
# Add to docker run command or write to .env file
```

Then create VM with:
```bash
gcloud compute instances create my-training-vm \
  --zone=europe-west1-b \
  --metadata-from-file=startup-script=startup-script.sh
```

## Option C: Build secrets into Docker image (LESS SECURE, but simpler)

Update cloudbuild.yaml to inject secrets during build:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    id: 'Build with secrets'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        docker build \
          --secret id=wandb,env=WANDB_API_KEY \
          -t europe-west1-docker.pkg.dev/.../images:train-latest \
          -f dockerfiles/train.dockerfile .
    secretEnv: ['WANDB_API_KEY']

availableSecrets:
  secretManager:
    - versionName: projects/proven-cosine-484310-s2/secrets/WANDB_API_KEY/versions/latest
      env: 'WANDB_API_KEY'
```

Then in Dockerfile, use:
```dockerfile
RUN --mount=type=secret,id=wandb \
    WANDB_API_KEY=$(cat /run/secrets/wandb) && export WANDB_API_KEY
```

**Recommendation:** Use Option A for Compute Engine - it's the most straightforward.
