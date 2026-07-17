#!/bin/bash
# Upload bag folders to BagArchive. RUN FROM THE LAB (wired) — bags are GBs;
# never do this over LTE tethering.
#   upload_bags.sh ~/bags/field_20260714_145901 [more...]
#   upload_bags.sh                 # no args: today's bags under ~/bags
set -u
A=http://203.250.35.87:31447
BAGS=("$@")
if [ ${#BAGS[@]} -eq 0 ]; then
  mapfile -t BAGS < <(ls -d ~/bags/*"$(date +%Y%m%d)"* 2>/dev/null)
  [ ${#BAGS[@]} -eq 0 ] && { echo "no bags for today under ~/bags — pass paths explicitly"; exit 1; }
fi

for D in "${BAGS[@]}"; do
  B=$(basename "$D")
  [ -d "$D" ] || { echo "skip $B (not a directory)"; continue; }
  args=()
  for f in "$D"/*; do args+=(-F "files=@$f" -F "paths=$B/$(basename "$f")"); done
  echo "== uploading $B ($(du -sh "$D" | cut -f1)) =="
  curl -s --max-time 7200 "${args[@]}" "$A/api/upload-folder"; echo
done

# index the new folders so they appear in the UI/API
curl -s -X POST "$A/api/scan" > /dev/null
sleep 12
echo "== archive status =="
for D in "${BAGS[@]}"; do
  B=$(basename "$D")
  curl -s "$A/api/bags?search=$B" | python3 -c "
import json, sys
for b in json.load(sys.stdin).get('items', []):
    print(f\"  {b['display_name']:34s} id={b['id']} health={b.get('health')} {b['size_bytes']/1e9:.1f}GB\")" 2>/dev/null
done
