"""Latency waterfall SVG for the round-5 report: one stacked bar per (video, run)."""
PHASES = [  # (label, color token)
    ('第一层 3D', '--p1'), ('能点、出卡片', '--p2'), ('全部有类型', '--p3'), ('模型全到', '--p4'), ('收尾', '--p5')]

def waterfall(lanes, t_max=None, w=900):
    """lanes: [(video, run, [t_first3d, t_cards, t_types, t_models, t_end])] in seconds."""
    t_max = t_max or max(ms[-1] for _, _, ms in lanes) * 1.05
    left, right, top, lane_h, gap = 190, 20, 34, 26, 16
    sx = lambda t: left + (w - left - right) * t / t_max
    rows, y, last_video = [], top, None
    for video, run, ms in lanes:
        if last_video is not None and video != last_video:
            y += gap
        if video != last_video:
            rows.append(f'<text x="8" y="{y + 17}" class="lv">{video}</text>')
        rows.append(f'<text x="{left - 8}" y="{y + 17}" class="lr" text-anchor="end">{run}</text>')
        t0 = 0.0
        for (label, tok), t in zip(PHASES, ms):
            rows.append(f'<rect x="{sx(t0):.1f}" y="{y}" width="{max(sx(t) - sx(t0), 0.5):.1f}" height="{lane_h - 6}" '
                        f'style="fill:var({tok})"><title>{label} {t:.0f} s</title></rect>')
            t0 = t
        for t in ms[1:4]:  # the milestones people ask about: clickable, typed, models
            rows.append(f'<text x="{sx(t):.1f}" y="{y - 3}" class="lt" text-anchor="middle">{t:.0f}</text>')
        last_video = video
        y += lane_h + 12
    ticks = ''.join(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 16}" y2="{y}" class="gl"/>'
                    f'<text x="{sx(t):.1f}" y="{y + 16}" class="lt" text-anchor="middle">{t} s</text>'
                    for t in range(0, int(t_max) + 1, 30))
    legend = ''.join(f'<rect x="{left + i * 140}" y="{y + 30}" width="12" height="12" style="fill:var({tok})"/>'
                     f'<text x="{left + i * 140 + 18}" y="{y + 41}" class="lr">{label}</text>'
                     for i, (label, tok) in enumerate(PHASES))
    h = y + 52
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="各阶段完成时间">'
            f'<style>.lv{{font:600 14px var(--f-body);fill:var(--ink)}}.lr{{font:12.5px var(--f-body);fill:var(--muted)}}'
            f'.lt{{font:11.5px var(--f-mono);fill:var(--muted)}}.gl{{stroke:var(--line);stroke-width:1}}</style>'
            f'{ticks}{"".join(rows)}{legend}</svg>')

def times_of(call):
    """r4b-style summary 'times' dict -> the 5 milestones (None when a stage is missing)."""
    t = call['times']
    pick = lambda *ks: next((t[k] for k in ks if isinstance(t.get(k), (int, float))), None)
    return [pick('first 3D'), pick('cards v1'), pick('types (densify pass)', 'all types'),
            pick('models final', 'Tier 1 models done', 'tier 1 done'), pick('call end')]

if __name__ == '__main__':
    import json
    d = json.load(open('/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/r4b-results/summary.json'))
    lanes = [(v, f'第 4 轮 {c}', times_of(d['videos'][v]['calls'][c])) for v in d['videos'] for c in ('warm', 'first')]
    assert all(None not in ms and ms == sorted(ms) for _, _, ms in lanes), lanes
    svg = waterfall(lanes)
    assert svg.count('<rect') == 6 * 5 + 5
    print(lanes[0], len(svg))
