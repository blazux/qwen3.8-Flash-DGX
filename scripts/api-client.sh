# Shared connection settings for flash and scripts/smoke-test.sh.
api_base() {
  if [ -n "${FLASH_API_BASE:-}" ]; then
    printf '%s\n' "${FLASH_API_BASE%/}"
  else
    docker inspect "$NAME" | python3 "$HERE/scripts/api-endpoint.py"
  fi
}
api_curl() {
  local ca="${FLASH_CA:-${CURL_CA_BUNDLE:-${CURL_CAINFO:-${SSL_CERT_FILE:-}}}}"
  local tls=()
  if [ "${FLASH_INSECURE:-0}" = 1 ]; then tls=(--insecure)
  elif [ -n "$ca" ]; then tls=(--cacert "$ca"); fi
  curl "${tls[@]}" "$@"
}
