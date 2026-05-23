#!/usr/bin/env bash
# Post-mortem: parse monitor.csv + HF Trainer log into HTML snippet,
# inject into gr00t_finetune_pick_orange.html §7 (Run telemetry).
#
# Run this on AutoDL after training finishes, then scp the snippet back to local
# and paste into the doc (or run with --inject to do it in-place if doc exists locally).
#
# Usage:
#   bash analyze_run.sh /root/autodl-tmp/monitor.csv \
#                       /root/autodl-tmp/.../logs/gr00t_n17_train_*.log \
#                       /root/autodl-tmp/.../outputs/gr00t-n17-leisaac-pick-orange \
#                       > run_telemetry_snippet.html

set -euo pipefail

CSV="${1:?usage: $0 <monitor.csv> <train.log> <output_dir>}"
TRAIN_LOG="${2:?missing train log}"
OUTPUT_DIR="${3:?missing output dir}"

[[ -f "$CSV" ]] || { echo "ERROR: $CSV not found" >&2; exit 1; }
[[ -f "$TRAIN_LOG" ]] || { echo "ERROR: $TRAIN_LOG not found" >&2; exit 1; }

PY=${PY:-/root/miniconda3/bin/python}

$PY - <<PYEOF
import csv, re, os, json, sys
from pathlib import Path

CSV = "$CSV"
TRAIN_LOG = "$TRAIN_LOG"
OUTPUT_DIR = "$OUTPUT_DIR"

# === parse monitor.csv ===
rows = []
with open(CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        for k in ('epoch_sec','gpu_util_pct','vram_used_mib','vram_total_mib','vram_pct','disk_used_gb','disk_pct'):
            try: r[k] = float(r[k]) if '.' in r[k] else int(r[k])
            except: r[k] = 0
        rows.append(r)

if not rows:
    print("<!-- monitor.csv empty -->", file=sys.stderr)
    sys.exit(0)

# === resource peaks / means ===
gpu_utils = [r['gpu_util_pct'] for r in rows if r['gpu_util_pct'] > 0]
vram_used = [r['vram_used_mib'] for r in rows]
vram_pct  = [r['vram_pct'] for r in rows]
disk_pct  = [r['disk_pct'] for r in rows]
vram_total = max(r['vram_total_mib'] for r in rows)

def mean(xs): return sum(xs)/len(xs) if xs else 0

stats = {
    'wall_sec': rows[-1]['epoch_sec'] - rows[0]['epoch_sec'],
    'gpu_peak': max(gpu_utils) if gpu_utils else 0,
    'gpu_mean': mean(gpu_utils),
    'vram_peak_mib': max(vram_used),
    'vram_peak_pct': max(vram_pct),
    'vram_mean_pct': mean(vram_pct),
    'vram_total_gib': vram_total / 1024,
    'disk_peak_pct': max(disk_pct),
    'disk_peak_gb': max(r['disk_used_gb'] for r in rows),
}

# === parse train log for loss curve + step throughput ===
losses = []
for line in open(TRAIN_LOG, errors='ignore'):
    # match HF Trainer dict logs: {'loss': 0.42, ..., 'step': 50}
    m = re.search(r"'loss':\s*([\d.eE+-]+).*?'step':\s*(\d+)", line)
    if m:
        losses.append((int(m.group(2)), float(m.group(1))))

# step throughput from monitor.csv (latest_step over wall_sec)
step_seq = [(r['epoch_sec'], r['latest_step']) for r in rows if str(r.get('latest_step','')).isdigit()]
if step_seq:
    t0, s0 = step_seq[0]
    t1, s1 = step_seq[-1]
    step_per_sec = (int(s1) - int(s0)) / max(1, t1 - t0)
else:
    step_per_sec = 0

# === SVG: loss curve ===
def svg_line(points, w=860, h=200, color='#cf222e', label='loss'):
    if not points: return ''
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax==xmin: xmax = xmin+1
    if ymax==ymin: ymax = ymin+1
    def sx(x): return 50 + (x-xmin)/(xmax-xmin) * (w-70)
    def sy(y): return h-30 - (y-ymin)/(ymax-ymin) * (h-50)
    path = 'M ' + ' L '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in points)
    return f'''<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
  <rect x="50" y="20" width="{w-70}" height="{h-50}" fill="none" stroke="#ccc"/>
  <text x="{w/2}" y="14" font-size="11" text-anchor="middle" font-weight="bold">{label}</text>
  <text x="10" y="{h/2}" font-size="9" transform="rotate(-90 10,{h/2})" text-anchor="middle">{ymin:.3f} ~ {ymax:.3f}</text>
  <text x="50" y="{h-10}" font-size="9">step {xmin}</text>
  <text x="{w-30}" y="{h-10}" font-size="9" text-anchor="end">step {xmax}</text>
  <path d="{path}" fill="none" stroke="{color}" stroke-width="1.5"/>
</svg>'''

# vram time series
vram_pts = [(r['epoch_sec']/60, r['vram_used_mib']/1024) for r in rows]
gpu_pts  = [(r['epoch_sec']/60, r['gpu_util_pct']) for r in rows]

loss_svg = svg_line(losses, label='train_loss vs step', color='#cf222e') if losses else '<p><i>no loss data in log</i></p>'
vram_svg = svg_line(vram_pts, label='VRAM (GiB) vs wall time (min)', color='#1f6feb')
gpu_svg  = svg_line(gpu_pts, label='GPU util % vs wall time (min)', color='#1a7f37')

# === ckpt eval scores (read from trainer_state.json in each ckpt dir) ===
ckpt_rows = []
if os.path.isdir(OUTPUT_DIR):
    for d in sorted(os.listdir(OUTPUT_DIR)):
        m = re.match(r'^checkpoint-(\d+)$', d)
        if not m: continue
        step = int(m.group(1))
        ts = os.path.join(OUTPUT_DIR, d, 'trainer_state.json')
        if not os.path.isfile(ts): continue
        state = json.load(open(ts))
        loss = None
        for e in state.get('log_history', []):
            if 'loss' in e and e.get('step') == step:
                loss = e['loss']
        ckpt_rows.append((step, loss))
ckpt_table_rows = '\n'.join(
    f'<tr><td>{s}</td><td>{l:.4f if l else "—"}</td><td>—</td><td>—</td><td>—</td><td>—</td><td>kept</td></tr>'
    for s, l in ckpt_rows
) or '<tr><td colspan="7"><i>no ckpts found</i></td></tr>'

# === emit HTML snippet ===
print(f"""<!-- generated by analyze_run.sh -->
<h3>7.1 Run metadata</h3>
<table>
<tr><th>项</th><th>值</th></tr>
<tr><td>wall time</td><td>{stats['wall_sec']/3600:.2f} hr ({int(stats['wall_sec'])} sec)</td></tr>
<tr><td>step throughput</td><td>{step_per_sec:.3f} step/s</td></tr>
<tr><td>total VRAM</td><td>{stats['vram_total_gib']:.0f} GiB</td></tr>
</table>

<h3>7.2 资源峰值 / 平均</h3>
<table>
<tr><th>指标</th><th>峰值</th><th>均值</th><th>结论</th></tr>
<tr><td>GPU util %</td><td>{stats['gpu_peak']:.0f}%</td><td>{stats['gpu_mean']:.0f}%</td>
    <td>{"<span class='tag-good'>✓ GPU 充分利用</span>" if stats['gpu_mean']>=90 else
        "<span class='tag-warn'>IO bottleneck，可调高 dataloader_num_workers</span>" if stats['gpu_mean']<70 else
        "<span class='tag-good'>正常</span>"}</td></tr>
<tr><td>VRAM (GiB / total)</td><td>{stats['vram_peak_mib']/1024:.1f} / {stats['vram_total_gib']:.0f} ({stats['vram_peak_pct']:.0f}%)</td>
    <td>{mean(vram_used)/1024:.1f} ({stats['vram_mean_pct']:.0f}%)</td>
    <td>{"<span class='tag-bad'>⚠️ &gt;90%, 下次降 batch</span>" if stats['vram_peak_pct']>90 else
        "<span class='tag-warn'>可升 batch (still &lt;70%)</span>" if stats['vram_peak_pct']<70 else
        "<span class='tag-good'>合理留余量</span>"}</td></tr>
<tr><td>disk %</td><td>{stats['disk_peak_pct']:.0f}% ({stats['disk_peak_gb']:.0f} GB)</td><td>—</td>
    <td>{"<span class='tag-bad'>⚠️ 磁盘紧</span>" if stats['disk_peak_pct']>90 else "<span class='tag-good'>prune 起效</span>"}</td></tr>
</table>

<h3>7.3 训练曲线</h3>
{loss_svg}
{vram_svg}
{gpu_svg}

<h3>7.4 Ckpt 列表</h3>
<table>
<tr><th>step</th><th>train_loss</th><th>eval rounds × ep</th><th>oranges / total</th><th>strict env</th><th>avg time</th><th>kept by prune?</th></tr>
{ckpt_table_rows}
</table>
""")
PYEOF
