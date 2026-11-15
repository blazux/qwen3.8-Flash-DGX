# Shared by flash setup/doctor/serve and direct scripts/serve.sh.
# Base labels alone cannot distinguish images built before/after patch 23.
image_has_dflash() {
  [ "$(docker image inspect -f '{{index .Config.Labels "qwen38.dflash"}}' "$IMAGE" 2>/dev/null)" = 1 ]
}

# Source guard succeeded at build time; the older validated overlay uses this same label.
image_has_split_prefill() {
  [ "$(docker image inspect -f '{{index .Config.Labels "dgx.prefix-boundaries"}}' "$IMAGE" 2>/dev/null)" = opt-in-v1 ]
}
validate_split_prefill() {
  case "${SPLIT_PREFILL:-0}" in
    0|1) return 0 ;;
    *) echo "!! SPLIT_PREFILL must be 0 or 1"; return 1 ;;
  esac
}
require_split_prefill() {
  validate_split_prefill || return 1
  [ "${SPLIT_PREFILL:-0}" = 0 ] && return 0
  image_has_split_prefill && return 0
  echo "!! $IMAGE lacks the prefix-boundary workaround — run ./flash setup prefix-cache-safe or rebuild with the prefix-boundary patch."
  return 1
}
