"""urllib companion to api_curl: use the same explicit TLS settings."""
import os
import ssl
import urllib.request


def urlopen(request, timeout):
    ca = next((os.environ[k] for k in
               ("FLASH_CA", "CURL_CA_BUNDLE", "CURL_CAINFO", "SSL_CERT_FILE")
               if os.environ.get(k)), None)
    context = (ssl._create_unverified_context() if os.environ.get("FLASH_INSECURE") == "1"
               else ssl.create_default_context(cafile=ca))
    return urllib.request.urlopen(request, timeout=timeout, context=context)
