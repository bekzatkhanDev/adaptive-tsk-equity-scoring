import re
lines=open('paper/sn-article-template/name.tex',encoding='utf-8').read().splitlines()
patterns=[r'211',r'191',r'186',r'34\%',r'33\.9',r'halving',r'6\{,\}000',r'5\{,\}993',r'asset-days',r'0\.16',r'1\.34',r'1\.22',r'0\.18',r'paid rows',r'dividends',r'26 paid',r'pre-registered',r'DSR',r'36--63',r'36\\$ trials']
for p in patterns:
    for i,l in enumerate(lines,1):
        for m in re.finditer(p,l):
            s=max(0,m.start()-70)
            print(f'{i}|{p}| ...{l[s:m.end()+80]}...')
