#!/bin/bash
# Check shared memory size

echo "=== Current Shared Memory Status ==="
df -h /dev/shm

echo -e "\n=== Shared Memory Usage ==="
du -sh /dev/shm/*

echo -e "\n=== Recommended Actions ==="
echo "Current size: $(df -h /dev/shm | tail -1 | awk '{print $2}')"
echo "If you need to increase shared memory:"
echo "  sudo mount -o remount,size=16G /dev/shm"
echo ""
echo "Or add to /etc/fstab:"
echo "  tmpfs /dev/shm tmpfs defaults,size=16G 0 0"
