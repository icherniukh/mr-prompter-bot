#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Run this script with sudo or as root." >&2
  exit 1
fi

if ! command -v nft >/dev/null 2>&1; then
  echo "nft command not found. Install nftables first." >&2
  exit 1
fi

NFT_CONF="/etc/nftables.conf"
BACKUP_PATH="/etc/nftables.conf.bak.$(date +%Y%m%d-%H%M%S)"
TMP_CONF="$(mktemp)"

cleanup() {
  rm -f "${TMP_CONF}"
}
trap cleanup EXIT

if [ -f "${NFT_CONF}" ]; then
  cp -f "${NFT_CONF}" "${BACKUP_PATH}"
  echo "Backed up ${NFT_CONF} to ${BACKUP_PATH}"
fi

cat >"${TMP_CONF}" <<'EOF'
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
  chain input {
    type filter hook input priority 0;
    policy drop;

    iif "lo" accept
    ct state established,related accept

    tcp dport 22 accept
    udp dport 60000-61000 accept

    ip protocol icmp accept
    ip6 nexthdr ipv6-icmp accept
  }

  chain forward {
    type filter hook forward priority 0;
    policy drop;
  }

  chain output {
    type filter hook output priority 0;
    policy accept;
  }
}
EOF

nft -c -f "${TMP_CONF}"
install -m 0644 "${TMP_CONF}" "${NFT_CONF}"
nft -f "${NFT_CONF}"
systemctl enable --now nftables

echo "nftables enabled with SSH (TCP 22) and Mosh (UDP 60000-61000) allowed."
