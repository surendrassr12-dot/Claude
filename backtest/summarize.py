import sys,json
rows=[json.loads(l) for l in sys.stdin if l.startswith('{')]
t=dict(trades=0,gross=0,costs=0,net=0); prof=0
for r in rows:
    for k in t: t[k]+=r.get(k,0)
    prof += r.get('net',0)>0
w=sum(r.get('win_pct',0)*r.get('trades',0) for r in rows)/max(t['trades'],1)
print(f"{sys.argv[1]:38} trades={t['trades']:4} win%={w:4.1f} gross={t['gross']:>8} costs={t['costs']:>7} net={t['net']:>8}  profitable stocks={prof}/{len(rows)}")
