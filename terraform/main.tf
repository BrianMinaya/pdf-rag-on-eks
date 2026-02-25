# =============================================================================
# main.tf -- OpenTofu Configuration & AWS Provider
# =============================================================================
#
# This file sets up two things:
#   1. The OpenTofu version and provider requirements
#   2. The AWS provider configuration
#
# Think of this as the "preamble" -- it tells OpenTofu what tools it needs
# (AWS provider) and what version of OpenTofu is required to run this code.
# =============================================================================

# ---------------------------------------------------------------------------
# OpenTofu Settings Block
# ---------------------------------------------------------------------------
# "required_version" ensures everyone uses a compatible version
# of OpenTofu. The ">= 1.5" constraint means OpenTofu 1.5 or newer.
#
# "required_providers" pins the AWS provider to major version 5.x. The "~>"
# operator means "any version >= 5.0 but < 6.0". This prevents breaking
# changes from a major version bump while still picking up bug fixes and
# new features within the 5.x line.
# ---------------------------------------------------------------------------
terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ---------------------------------------------------------------------------
# AWS Provider
# ---------------------------------------------------------------------------
# The provider block configures how OpenTofu talks to AWS. The region
# determines which AWS data center your resources are created in.
#
# We pull the region from a variable so it can be changed without editing
# this file (see variables.tf). Default is us-east-1 (Northern Virginia),
# which has the broadest GPU instance availability.
#
# Authentication is NOT configured here -- OpenTofu picks up credentials
# from your environment (AWS CLI profile, env vars, or IAM role). This is
# a security best practice: never hardcode credentials in code.
# ---------------------------------------------------------------------------
provider "aws" {
  region = var.region
}
