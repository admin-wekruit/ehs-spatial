import sys,re,base64; S=sys.argv[1]; sys.path.insert(0,S)
from latency import waterfall
svg=waterfall([('ME340','第 4 轮 热启动',[18.0,43.1,97.8,182.8,230.2]),('ME340','第 5 轮 热启动',[15.9,37.4,89.6,266.5,266.8]),('ME340','第 5 轮 首次调用',[20.3,38.6,94.0,260.9,261.2])])
sec=open(f'{S}/me340-run2.html',encoding='utf-8').read().replace('{{svg}}',svg)
p=f'{S}/body.html'; t=open(p,encoding='utf-8').read()
i0=t.index('<section id="me340">'); i1=t.index('</section>',i0)+len('</section>')
t=t[:i0]+sec+t[i1:]
t=t.replace('<span class="chip warn">ME340 合并版：第 1 次完成</span><span class="chip run">第 2 次（修正后）：进行中</span>',"<span class=\"chip good\">ME340 合并版：完成</span><span class=\"chip run\">Sam's Club、Walmart：进行中</span>")
t=re.sub(r'更新于 2026-09-29 \d\d:\d\d','更新于 2026-09-29 23:40',t)
open(p,'w',encoding='utf-8').write(t)
full=open(f'{S}/head.html',encoding='utf-8').read()+t; open(f'{S}/template.html','w',encoding='utf-8').write(full)
out=re.sub(r'\{\{img:([a-z0-9]+)\}\}',lambda m:'data:image/jpeg;base64,'+base64.b64encode(open(f'{S}/img/{m.group(1)}.jpg','rb').read()).decode(),full)
open(f'{S}/panoptes-round5.html','w',encoding='utf-8').write(out); print('ok',len(out)//1024,'KB', out.count('{{img'), 'ME340 合并版：完成' in out)
