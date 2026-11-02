# Shared by flash setup/doctor/serve and direct scripts/serve.sh.
# Base labels alone cannot distinguish images built before/after patch 23.
image_has_dflash() {
  [ "$(docker image inspect -f '{{index .Config.Labels "qwen38.dflash"}}' "$IMAGE" 2>/dev/null)" = 1 ]
}
