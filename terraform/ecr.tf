# =============================================================================
# ecr.tf -- Elastic Container Registry (ECR) Repositories
# =============================================================================
#
# ECR is AWS's private Docker image registry. Instead of pulling images from
# Docker Hub (which would require internet egress and has rate limits), we
# store our application images in ECR within our own AWS account.
#
# We create two repositories for our custom-built services:
#
#   1. chat-api   -- The FastAPI service that handles user questions,
#                    retrieves context from Qdrant, and calls vLLM
#
#   2. ingestion  -- The batch job that pulls PDFs, chunks text, generates
#                    embeddings, and stores vectors in Qdrant
#
# Other services (vLLM, TEI embedding, Qdrant, PostgreSQL) use official
# Docker images pulled directly from their public registries.
#
# Each repository is configured with:
#   - image_tag_mutability = "MUTABLE": allows overwriting tags like "latest".
#     This simplifies the development workflow (push with same tag, k8s pulls
#     new image). In production you would use IMMUTABLE tags for traceability.
#
#   - force_delete = true: allows Terraform to delete the repo even if it
#     contains images. Important for cleanup -- without this, you would
#     have to manually delete all images before destroying the infrastructure.
#
#   - scan_on_push = true: automatically scans images for known CVEs
#     (security vulnerabilities) every time you push. Free and catches
#     issues like outdated base images with known exploits.
# =============================================================================

# ---------------------------------------------------------------------------
# Local variable to define all repository names in one place.
# Using a local avoids repeating the project_name prefix in each resource
# and makes it easy to add or remove repositories.
# ---------------------------------------------------------------------------
locals {
  ecr_repositories = {
    chat_api  = "${var.project_name}/chat-api"
    ingestion = "${var.project_name}/ingestion"
  }
}

# ---------------------------------------------------------------------------
# ECR Repositories
# ---------------------------------------------------------------------------
# "for_each" creates one aws_ecr_repository resource per entry in the
# locals map above. This is cleaner than writing four separate resource
# blocks with identical configuration. The "each.key" (e.g., "chat_api")
# becomes the Terraform resource identifier, and "each.value" (e.g.,
# "pdf-rag-chatbot/chat-api") becomes the repository name in ECR.
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "repos" {
  for_each = local.ecr_repositories

  name = each.value

  # MUTABLE allows pushing a new image with the same tag (e.g., "latest").
  # Convenient for development; use IMMUTABLE in production for auditability.
  image_tag_mutability = "MUTABLE"

  # Allow Terraform to delete the repository even if it still contains images.
  # Without this, "terraform destroy" would fail if any images have been pushed.
  force_delete = true

  # Automatically scan every pushed image for known security vulnerabilities.
  # Results appear in the ECR console and can trigger alerts via EventBridge.
  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
}
