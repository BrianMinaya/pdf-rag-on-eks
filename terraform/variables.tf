# =============================================================================
# variables.tf -- Input Variables
# =============================================================================
#
# Variables are the "knobs" you can turn to customize this deployment without
# changing the actual resource definitions. Each variable has:
#   - description: what it controls (shown in CLI prompts and docs)
#   - type:        the data type Terraform expects
#   - default:     the value used if you don't explicitly set one
#
# You can override defaults in terraform.tfvars, via CLI flags (-var), or
# through environment variables (TF_VAR_<name>).
# =============================================================================

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

variable "region" {
  description = "AWS region to deploy all resources into. us-east-1 is chosen because it has the widest GPU instance availability and lowest spot prices."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix applied to every resource for easy identification. Used in resource names, tags, and EKS cluster naming."
  type        = string
  default     = "pdf-rag-chatbot"
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "The IP address range for the entire VPC, in CIDR notation. A /16 gives us 65,536 IP addresses, which is more than enough and leaves room for growth."
  type        = string
  default     = "10.0.0.0/16"
}

# ---------------------------------------------------------------------------
# EKS (Kubernetes)
# ---------------------------------------------------------------------------

variable "eks_version" {
  description = "Kubernetes version for the EKS control plane. Should match the version you have tested against. AWS manages the control plane upgrades, but you choose when to move to a new version."
  type        = string
  default     = "1.31"
}

variable "cpu_instance_type" {
  description = "EC2 instance type for CPU worker nodes. t3.xlarge is an x86 instance with 4 vCPUs and 16 GB RAM -- enough to run TEI (Nomic V1.5) alongside Qdrant and PostgreSQL. Supports all official Docker images without cross-compilation."
  type        = string
  default     = "t3.xlarge"
}

variable "gpu_instance_types" {
  description = "List of EC2 instance types for the GPU node group. Multiple types are specified to improve spot instance availability -- if g4dn.xlarge is unavailable, EKS will try g4dn.2xlarge. Both have NVIDIA T4 GPUs suitable for running Llama 3.1 8B with vLLM."
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

variable "gpu_min_size" {
  description = "Minimum number of GPU nodes. Set to 0 so the GPU node group can scale to zero when not in use, saving significant cost (GPU instances are expensive)."
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  description = "Maximum number of GPU nodes. Set to 1 since we only need a single GPU for Llama 3.1 8B inference."
  type        = number
  default     = 1
}

variable "gpu_desired_size" {
  description = "Initial number of GPU nodes when the cluster is created. Set to 0 to avoid paying for GPU compute until you actually need it. Use startup.sh to scale up when ready to run inference."
  type        = number
  default     = 0
}
