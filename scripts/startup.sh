#!/usr/bin/env bash
# =============================================================================
# Startup -- Scale ALL node groups back up
#
# Scales both CPU and GPU node groups to their desired running sizes:
#   - CPU nodes: desiredSize=1 (t3.xlarge, on-demand, ~$0.17/hr)
#   - GPU nodes: desiredSize=1 (g4dn.xlarge, spot, ~$0.21/hr)
#
# Typical startup time:
#   - CPU node: ~90 seconds to join cluster, pods auto-reschedule
#   - GPU node: ~90 seconds to join + ~3-5 min for vLLM model loading
#
# Usage: ./scripts/startup.sh
# =============================================================================
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-pdf-rag-chatbot}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# -----------------------------------------------------------------------
# Look up node group names dynamically (they have Terraform-generated
# timestamp suffixes that change when node groups are recreated).
# -----------------------------------------------------------------------
echo "=== Starting All Nodes ==="
echo "Cluster: ${CLUSTER_NAME}"
echo "Region:  ${AWS_REGION}"
echo ""

echo ">>> Looking up node group names..."
NODEGROUPS=$(aws eks list-nodegroups \
  --cluster-name "${CLUSTER_NAME}" \
  --region "${AWS_REGION}" \
  --query 'nodegroups[]' \
  --output text)

if [ -z "${NODEGROUPS}" ]; then
  echo "No node groups found. Nothing to start."
  exit 0
fi

echo "Found node groups: ${NODEGROUPS}"
echo ""

# -----------------------------------------------------------------------
# Scale each node group to its desired running size.
# CPU nodes: 1 instance (maxSize=2 for headroom)
# GPU nodes: 1 instance (maxSize=1)
# -----------------------------------------------------------------------
for NG in ${NODEGROUPS}; do
  if [[ "${NG}" == gpu_nodes* ]]; then
    MAX_SIZE=1
    DESIRED=1
    LABEL="GPU"
  else
    MAX_SIZE=2
    DESIRED=1
    LABEL="CPU"
  fi

  echo ">>> Scaling ${LABEL} node group (${NG}) to ${DESIRED}..."
  aws eks update-nodegroup-config \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name "${NG}" \
    --scaling-config minSize=0,maxSize="${MAX_SIZE}",desiredSize="${DESIRED}" \
    --region "${AWS_REGION}" \
    --query 'update.status' \
    --output text
  echo ""
done

echo "=== All node groups scaling up ==="
echo ""
echo "Typical wait times:"
echo "  CPU node: ~90 seconds (pods auto-reschedule)"
echo "  GPU node: ~90 seconds + 3-5 min for vLLM model loading"
echo ""
echo "Monitor progress:"
echo "  kubectl get nodes -w"
echo "  kubectl -n pdf-rag-chatbot get pods -w"
