import sys,re,base64,json; S=sys.argv[1]; sys.path.insert(0,S)
from latency import waterfall
q=json.load(open('/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/r5b-results/samsclub-a2/quick.json'))
tw=q['times_warm']; tf=q['times_first']
lane=lambda t:[t['first 3D'],t['cards v1'],t['types (densify pass)'],t['all generated models in (models final)'],t['call end']]
svg=waterfall([("Sam's Club",'第 4 轮 热启动',[15.3,30.1,87.5,180.3,228.5]),("Sam's Club",'第 5 轮 热启动',lane(tw)),("Sam's Club",'第 5 轮 首次调用',lane(tf))])
f=lambda v:f'{v:.1f}'
sams=open(f'{S}/sams.html',encoding='utf-8').read().replace('{{svg}}',svg)
for k,v in {'t_first3d':tw['first 3D'],'t_cards':tw['cards v1'],'t_surf':tw['tier 0 surfaces v1'],'t_types':tw['types (densify pass)'],'t_gen1':tw['first generated model'],'t_genall':tw['all generated models in (models final)']}.items():
    sams=sams.replace('{{'+k+'}}',f(v))
issues=open(f'{S}/issues.html',encoding='utf-8').read()
p=f'{S}/body.html'; t=open(p,encoding='utf-8').read()
for sid in ('issues','position','samsclub'):
    if f'<section id="{sid}">' in t:
        i0=t.index(f'<section id="{sid}">'); i1=t.index('</section>',i0)+len('</section>'); t=t[:i0]+t[i1:].lstrip('\n')
i=t.index('<section id="me340">'); t=t[:i]+issues+t[i:]
j=t.index('</section>',t.index('<section id="me340">'))+len('</section>'); t=t[:j]+'\n\n'+sams+t[j:]
t=t.replace("<span class=\"chip run\">Sam's Club、Walmart：进行中</span>","<span class=\"chip good\">Sam's Club：完成</span><span class=\"chip run\">Walmart：进行中</span>")
t=re.sub(r'更新于 2026-09-29 \d\d:\d\d','更新于 2026-09-29 23:50',t)
open(p,'w',encoding='utf-8').write(t)
full=open(f'{S}/head.html',encoding='utf-8').read()+t; open(f'{S}/template.html','w',encoding='utf-8').write(full)
out=re.sub(r'\{\{img:([a-z0-9]+)\}\}',lambda m:'data:image/jpeg;base64,'+base64.b64encode(open(f'{S}/img/{m.group(1)}.jpg','rb').read()).decode(),full)
open(f'{S}/panoptes-round5.html','w',encoding='utf-8').write(out)
print('ok',len(out)//1024,'KB','left',out.count('{{'), [t.count(f'<section id="{x}">') for x in ('issues','position','me340','samsclub')])
