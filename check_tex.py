import re

t = open('paper/sn-article-template/name.tex', encoding='utf-8').read()

# labels / refs / citations
L = re.findall(r'\\label\{([^}]+)\}', t)
R = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', t)
C = [x for c in re.findall(r'\\cite\{([^}]+)\}', t) for x in c.split(',')]
B = re.findall(r'\\bibitem\{([^}]+)\}', t)
print('ref-missing:', sorted(set(R) - set(L)))
print('cited-not-defined:', sorted(set(C) - set(B)))
print('defined-not-cited:', sorted(set(B) - set(C)))
print('dup-labels:', sorted({x for x in L if L.count(x) > 1}))

# brace balance, comments stripped, \{ \} ignored
tn = re.sub(r'(?<!\\)%[^\n]*', '', t)
tn = tn.replace(r'\{', '').replace(r'\}', '')
d, bad = 0, False
for ch in tn:
    if ch == '{':
        d += 1
    elif ch == '}':
        d -= 1
        if d < 0:
            bad = True
            break
print('brace:', 'OK' if d == 0 and not bad else f'BAD depth={d} bad={bad}')

# environments
for e in ['equation', 'align', 'table', 'figure', 'proposition', 'proof',
          'theorem', 'remark', 'algorithm', 'enumerate', 'itemize']:
    a = t.count('\\begin{' + e + '}')
    b = t.count('\\end{' + e + '}')
    if a != b:
        print('ENV MISMATCH', e, a, b)
print('envs checked')

# algorithmic balance
ai = len(re.findall(r'\\If\{', t)); ae = len(re.findall(r'\\EndIf', t))
af = len(re.findall(r'\\For\{', t)); aft = len(re.findall(r'\\EndFor', t))
print(f'algorithmic If={ai}/{ae} For={af}/{aft}')

# abstract word count
m = re.search(r'\\abstract\{(.*?)\n\}', t, re.S)
if m:
    w = re.findall(r"[A-Za-z0-9']+", re.sub(r'\\[a-zA-Z]+|[{}$%~]', ' ', m.group(1)))
    print('abstract words:', len(w))

# leftover stale claims
for s in ['gain comes from per-row', 'Anti-Frontrunning', 'tightest natural bound',
          'fivefold', '1.71']:
    n = t.count(s)
    if n:
        print('STALE:', s, n)
print('stale scan done')
