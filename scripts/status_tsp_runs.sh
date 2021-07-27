echo running:
bash scripts/ts_command.sh| grep runn | wc | cut -d' ' -f2-10
echo finish:
bash scripts/ts_command.sh| grep finish | wc | cut -d' ' -f2-10
echo queue:
bash scripts/ts_command.sh| grep queu | wc | cut -d' ' -f2-10