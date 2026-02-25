# =============================================================================
# outputs.tf -- Values Exported After "terraform apply"
# =============================================================================
#
# Outputs are values that Terraform prints after a successful apply. They
# serve two purposes:
#   1. Display useful information (like the cluster endpoint URL)
#   2. Allow other Terraform configurations to reference these values
#      (if you split infrastructure across multiple Terraform projects)
#
# After running "terraform apply", you will see these values printed. You
# can also retrieve them anytime with "terraform output" or
# "terraform output <name>".
# =============================================================================

# ---------------------------------------------------------------------------
# EKS Cluster Outputs
# ---------------------------------------------------------------------------

output "cluster_name" {
  description = "Name of the EKS cluster. Used by kubectl, helm, and CI/CD pipelines to identify which cluster to interact with."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "The URL of the Kubernetes API server. This is where kubectl sends commands. The endpoint is publicly accessible (see eks.tf) so you can manage the cluster from your laptop."
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64-encoded certificate for the cluster's Certificate Authority. kubectl uses this to verify it is talking to the real cluster and not an impersonator (TLS verification). Automatically configured by 'aws eks update-kubeconfig'."
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

# ---------------------------------------------------------------------------
# ECR Repository URLs
# ---------------------------------------------------------------------------

output "ecr_repository_urls" {
  description = "Map of ECR repository names to their URLs. Use these URLs when tagging and pushing Docker images. Example: docker tag my-image:latest <url>:latest && docker push <url>:latest"
  value = {
    for key, repo in aws_ecr_repository.repos : key => repo.repository_url
  }
}

# ---------------------------------------------------------------------------
# VPC Output
# ---------------------------------------------------------------------------

output "vpc_id" {
  description = "ID of the VPC. Useful if you need to create additional resources (like an RDS database) in the same VPC later."
  value       = module.vpc.vpc_id
}

# ---------------------------------------------------------------------------
# Helper Command
# ---------------------------------------------------------------------------

output "configure_kubectl" {
  description = "Run this command to configure kubectl to talk to the new EKS cluster. It updates your ~/.kube/config with the cluster endpoint, certificate, and authentication details."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}
