import re,os
for f in ['README.md','data/README.md']:
    if not os.path.exists(f): continue
    lines=open(f,encoding='utf-8').read().splitlines()
    for p in [r'6,?000',r'5,?993',r'78\%',r'2\.58',r'0\.484',r'48\.4',r'38\.5',r'191',r'2025-05-25',r'45',r'dividend']:
        for i,l in enumerate(lines,1):
            for m in re.finditer(p,l):
                s=max(0,m.start()-60); print(f'{f}:{i}|{p}| ...{l[s:m.end()+70]}...')
