# =============================================================================
# vpc.tf -- Virtual Private Cloud (Network Foundation)
# =============================================================================
#
# A VPC is your own isolated network inside AWS. Everything we deploy -- the
# EKS cluster, worker nodes, load balancers -- lives inside this VPC.
#
# Network layout:
#   VPC: 10.0.0.0/16 (65,536 IPs)
#   ├── Private Subnet AZ-a: 10.0.1.0/24   (256 IPs, worker nodes live here)
#   ├── Private Subnet AZ-b: 10.0.2.0/24   (256 IPs, worker nodes live here)
#   ├── Public Subnet AZ-a:  10.0.101.0/24  (256 IPs, load balancers, NAT GW)
#   └── Public Subnet AZ-b:  10.0.102.0/24  (256 IPs, load balancers)
#
# Why two Availability Zones?
#   AWS best practice for resilience. If one data center (AZ) has issues,
#   workloads can run in the other. EKS requires at least 2 AZs.
#
# Why private + public subnets?
#   - Private subnets have NO direct internet access. Worker nodes go here
#     so they are not directly reachable from the internet (security).
#   - Public subnets have internet-facing resources like load balancers and
#     the NAT Gateway (which lets private nodes reach the internet for
#     pulling container images, etc.).
#
# NAT Gateway:
#   A NAT (Network Address Translation) Gateway lets resources in private
#   subnets make outbound internet requests (e.g., pulling Docker images)
#   without being directly accessible from the internet. We use a single
#   NAT Gateway to save cost (~$32/month per gateway). In production you
#   would use one per AZ for high availability.
# =============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  # ---------------------------------------------------------------------------
  # Basic VPC settings
  # ---------------------------------------------------------------------------
  name = var.project_name
  cidr = var.vpc_cidr

  # Deploy across two Availability Zones for resilience.
  # AZs are physically separate data centers within the same region.
  azs = ["${var.region}a", "${var.region}b"]

  # ---------------------------------------------------------------------------
  # Subnet CIDRs
  # ---------------------------------------------------------------------------
  # Private subnets: where EKS worker nodes (pods) run. No direct internet
  # ingress, which is a security best practice.
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  # Public subnets: where internet-facing load balancers and the NAT Gateway
  # are placed. These subnets get public IP addresses.
  public_subnets = ["10.0.101.0/24", "10.0.102.0/24"]

  # ---------------------------------------------------------------------------
  # NAT Gateway (allows private subnet outbound internet access)
  # ---------------------------------------------------------------------------
  # enable_nat_gateway: creates the NAT Gateway resource
  # single_nat_gateway: uses ONE NAT Gateway shared across all AZs instead of
  #   one per AZ. This saves ~$32/month but means if that AZ goes down,
  #   private subnets in the other AZ lose outbound internet. Acceptable
  #   trade-off for a non-critical workload.
  enable_nat_gateway = true
  single_nat_gateway = true

  # ---------------------------------------------------------------------------
  # Tags for EKS subnet auto-discovery
  # ---------------------------------------------------------------------------
  # EKS and the AWS Load Balancer Controller use these tags to automatically
  # find the right subnets when creating load balancers:
  #
  # "kubernetes.io/cluster/<name>" = "shared"
  #   Tells EKS these subnets belong to (or are shared with) our cluster.
  #
  # "kubernetes.io/role/internal-elb" = "1"
  #   Marks private subnets for internal (cluster-only) load balancers.
  #
  # "kubernetes.io/role/elb" = "1"
  #   Marks public subnets for internet-facing load balancers.
  # ---------------------------------------------------------------------------
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"               = "1"
    "kubernetes.io/cluster/${var.project_name}" = "shared"
  }

  public_subnet_tags = {
    "kubernetes.io/role/elb"                         = "1"
    "kubernetes.io/cluster/${var.project_name}" = "shared"
  }

  # Common tags applied to every resource created by this module.
  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
}
