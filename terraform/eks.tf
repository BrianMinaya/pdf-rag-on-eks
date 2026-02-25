# =============================================================================
# eks.tf -- Elastic Kubernetes Service (EKS) Cluster
# =============================================================================
#
# EKS is AWS's managed Kubernetes service. AWS handles the control plane
# (API server, etcd, scheduler) while we manage the worker nodes that
# actually run our application pods.
#
# Our cluster has TWO node groups:
#
#   1. CPU Nodes (t3.xlarge, x86, ON-DEMAND)
#      - Run: Chat API, Embedding Server, Qdrant, PostgreSQL, Ingestion
#      - Always running (min 1 node) to keep the system available
#
#   2. GPU Nodes (g4dn.xlarge/2xlarge, NVIDIA T4, SPOT)
#      - Run: vLLM (Llama 3.1 8B inference)
#      - Scale to 0 when not needed to save cost
#      - Use spot instances (up to 70% cheaper than on-demand)
#      - Tainted so ONLY GPU workloads get scheduled here
#
# Why spot instances for GPU?
#   GPU instances are expensive ($0.526/hr for g4dn.xlarge on-demand).
#   Spot instances use spare AWS capacity at up to 70% discount. The
#   trade-off is AWS can reclaim them with 2 minutes notice. For a non-critical
#   workload this is acceptable -- a brief interruption while a new spot
#   instance spins up. Multiple instance types improve spot availability.
#
# Why taints on GPU nodes?
#   A "taint" is like a "keep out" sign. The taint on GPU nodes prevents
#   regular pods (Qdrant, PostgreSQL, etc.) from being scheduled there.
#   Only pods that explicitly "tolerate" the taint (vLLM) will run on
#   GPU nodes. This ensures expensive GPU resources are reserved for
#   workloads that actually need them.
# =============================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  # ---------------------------------------------------------------------------
  # Cluster basics
  # ---------------------------------------------------------------------------
  cluster_name    = var.project_name
  cluster_version = var.eks_version

  # Place the EKS control plane and worker nodes in our VPC.
  # Worker nodes go in private subnets (no direct internet exposure).
  # NOTE: subnet_ids here apply to the EKS control plane ENIs and act as the
  # default for node groups that don't specify their own subnet_ids.
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # ---------------------------------------------------------------------------
  # Cluster access
  # ---------------------------------------------------------------------------
  # Allow kubectl access from the internet (your laptop). In production you
  # would restrict this to a VPN or bastion host CIDR, but for development we
  # keep it open so developers can easily interact with the cluster.
  cluster_endpoint_public_access = true

  # This grants the IAM principal running "terraform apply" full admin access
  # to the EKS cluster. Without this, you would be locked out of your own
  # cluster after creation (a common gotcha with EKS).
  enable_cluster_creator_admin_permissions = true

  # ---------------------------------------------------------------------------
  # Cluster Addons
  # ---------------------------------------------------------------------------
  # Addons are AWS-managed components that run inside the cluster. They are
  # kept up to date by AWS and are the recommended way to install core
  # Kubernetes functionality on EKS.
  #
  # coredns:            Cluster DNS -- lets pods find each other by name
  #                     (e.g., "qdrant" resolves to the Qdrant pod IP)
  #
  # kube-proxy:         Network rules on each node that route traffic to the
  #                     correct pod. Required for Kubernetes Services to work.
  #
  # vpc-cni:            The AWS VPC CNI plugin assigns real VPC IP addresses
  #                     to each pod, so pods can communicate directly without
  #                     overlay networking. Better performance than alternatives.
  #
  # aws-ebs-csi-driver: REQUIRED for Persistent Volume Claims (PVCs). Without
  #                     this, pods cannot request persistent EBS storage. We
  #                     need this for Qdrant (vector data) and PostgreSQL
  #                     (application data) to survive pod restarts.
  #                     The EBS CSI driver needs IAM permissions to create and
  #                     attach EBS volumes, which we provide via a service
  #                     account role (IRSA -- IAM Roles for Service Accounts).
  # ---------------------------------------------------------------------------
  cluster_addons = {
    coredns = {
      # "most_recent = true" tells the module to use the latest addon version
      # compatible with our cluster version. This is the simplest approach and
      # keeps addons up to date.
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true

      # The EBS CSI driver needs IAM permissions to manage EBS volumes on our
      # behalf. This creates an IAM role with the AmazonEBSCSIDriverPolicy
      # attached and binds it to the driver's Kubernetes service account via
      # IRSA (IAM Roles for Service Accounts).
      #
      # IRSA is an EKS feature that lets individual pods assume specific IAM
      # roles, following the principle of least privilege. Only the EBS CSI
      # driver pod gets volume management permissions -- no other pod does.
      service_account_role_arn = module.ebs_csi_irsa_role.iam_role_arn
    }
  }

  # ---------------------------------------------------------------------------
  # Managed Node Groups
  # ---------------------------------------------------------------------------
  # Node groups are pools of EC2 instances that run your Kubernetes pods.
  # "Managed" means AWS handles node provisioning, patching, and draining
  # during updates. You define the instance type, scaling limits, and labels.
  # ---------------------------------------------------------------------------
  eks_managed_node_groups = {

    # -------------------------------------------------------------------------
    # CPU Node Group
    # -------------------------------------------------------------------------
    # These nodes run everything EXCEPT the LLM. The t3.xlarge is an x86
    # instance: 4 vCPUs, 16 GB RAM, ~$0.1664/hr. We use x86 to match the
    # production environments and ensures compatibility with all official Docker
    # images (e.g., HuggingFace TEI which only ships x86 builds).
    #
    # ami_type "AL2023_x86_64_STANDARD" tells EKS to use the Amazon Linux 2023
    # AMI built for x86 processors. Docker images must be built for
    # linux/amd64 to run on these nodes.
    #
    # Labels are key-value pairs attached to nodes. Pods use "nodeSelector" or
    # "nodeAffinity" in their spec to request scheduling on nodes with
    # specific labels. Our CPU workloads will have:
    #   nodeSelector:
    #     workload: cpu
    # -------------------------------------------------------------------------
    cpu_nodes = {
      instance_types = [var.cpu_instance_type]
      ami_type       = "AL2023_x86_64_STANDARD"

      # PIN TO A SINGLE AZ
      # EBS volumes (used by Qdrant and PostgreSQL StatefulSets) are locked to
      # the Availability Zone where they were first created. If the CPU node
      # launches in a different AZ, the pods cannot attach their volumes and
      # will be stuck in Pending. By restricting the CPU node group to a single
      # private subnet (us-east-1b / 10.0.2.0/24), every new node is guaranteed
      # to land in the same AZ as the existing EBS volumes.
      #
      # [slice(list, start, end)] extracts a portion of a list:
      #   slice(private_subnets, 1, 2)  =>  just the second subnet (us-east-1b)
      subnet_ids = slice(module.vpc.private_subnets, 1, 2)

      min_size     = 1
      max_size     = 2
      desired_size = 1

      # ON_DEMAND instances are always available (no interruptions). We use
      # on-demand for CPU nodes because they run stateful workloads (Qdrant,
      # PostgreSQL) that should not be interrupted.
      capacity_type = "ON_DEMAND"

      labels = {
        workload = "cpu"
      }
    }

    # -------------------------------------------------------------------------
    # GPU Node Group
    # -------------------------------------------------------------------------
    # These nodes run vLLM (Llama 3.1 8B inference). The g4dn instances have
    # NVIDIA T4 GPUs with 16 GB VRAM, enough for Llama 3.1 8B in INT4
    # quantization (~5 GB model + KV cache).
    #
    # ami_type "AL2_x86_64_GPU" includes the NVIDIA drivers pre-installed.
    # GPU instances are x86_64 (not ARM), so vLLM images must be built for
    # linux/amd64.
    #
    # SPOT instances use spare AWS capacity at a steep discount. We list
    # multiple instance types so if one type is unavailable, EKS will try
    # the next. The trade-off: AWS can reclaim spot instances with 2 minutes
    # notice. For a non-critical workload, a brief LLM outage is acceptable.
    #
    # desired_size = 0 means NO GPU nodes at creation time. This saves cost
    # until you are ready to deploy vLLM. Scale up with:
    #   aws eks update-nodegroup-config \
    #     --cluster-name pdf-rag-chatbot \
    #     --nodegroup-name gpu_nodes \
    #     --scaling-config desiredSize=1
    #
    # The taint prevents non-GPU workloads from landing on these expensive
    # nodes. Only pods with a matching toleration will be scheduled here:
    #   tolerations:
    #     - key: nvidia.com/gpu
    #       operator: Equal
    #       value: "true"
    #       effect: NoSchedule
    # -------------------------------------------------------------------------
    gpu_nodes = {
      instance_types = var.gpu_instance_types
      ami_type       = "AL2_x86_64_GPU"

      min_size     = var.gpu_min_size
      max_size     = var.gpu_max_size
      desired_size = var.gpu_desired_size

      capacity_type = "SPOT"

      # -----------------------------------------------------------------------
      # ROOT VOLUME SIZE
      # -----------------------------------------------------------------------
      # The default EKS root volume is ~20GB, which is NOT enough for GPU
      # workloads. The vLLM Docker image alone is ~15GB, plus the NVIDIA
      # device plugin, CUDA libraries, and model cache. 100GB provides
      # comfortable headroom for image pulls and model downloads.
      #
      # IMPORTANT: Must use block_device_mappings, NOT disk_size. The EKS
      # module creates a launch template for each node group, and when a
      # launch template exists, the disk_size parameter on the node group
      # is ignored. The block_device_mappings go into the launch template.
      # -----------------------------------------------------------------------
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            delete_on_termination = true
          }
        }
      }

      labels = {
        workload = "gpu"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  # Common tags applied to all EKS-managed resources.
  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
}

# =============================================================================
# IAM Role for the EBS CSI Driver (IRSA)
# =============================================================================
#
# IRSA (IAM Roles for Service Accounts) lets a Kubernetes pod assume a
# specific IAM role. This is the AWS-recommended way to grant permissions
# to pods -- much more secure than attaching broad policies to the node
# instance role.
#
# Here we create an IAM role that:
#   1. Can only be assumed by the "ebs-csi-controller-sa" service account
#      in the "kube-system" namespace (least privilege)
#   2. Has the AmazonEBSCSIDriverPolicy attached, which grants permissions
#      to create, attach, detach, and delete EBS volumes
#
# The EKS module wires this up via OIDC (OpenID Connect), which establishes
# a trust relationship between the Kubernetes cluster and AWS IAM.
# =============================================================================
module "ebs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.project_name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
}
