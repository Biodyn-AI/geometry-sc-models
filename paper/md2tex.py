#!/usr/bin/env python3
"""Convert supplement.md -> supplement.tex via pandoc, with unicode/math handling."""
import re, subprocess, sys, os

SRC = 'supplement.md'
OUT = 'supplement_body.tex'

s = open(SRC).read()

# ---------------------------------------------------------------- 1. multi-char
# Specific formulas and compound symbols, done before the single-char map.
SPECIFIC = [
    ("`x·EEᵀ`",               "@@M@@x\\cdot EE^{\\mathsf{T}}@@M@@"),
    ("t̂", "@@M@@\\hat{t}@@M@@"),
    ("‖D‖",                   "@@M@@\\lVert D\\rVert@@M@@"),
    ("θ_far",                      "@@M@@\\theta_{\\mathrm{far}}@@M@@"),
    ("H_flat",                          "@@M@@H_{\\mathrm{flat}}@@M@@"),
    ("R_diff",                          "@@M@@R_{\\mathrm{diff}}@@M@@"),
    ("s_ery²",                     "@@M@@s_{\\mathrm{ery}}^{2}@@M@@"),
    ("ΔR_A",                       "@@M@@\\Delta R_A@@M@@"),
    ("ΔR_B",                       "@@M@@\\Delta R_B@@M@@"),
    ("val_corr",                        "`val_corr`"),
    ("HSC_1",                           "@@M@@\\mathrm{HSC}_1@@M@@"),
    ("Ery_2",                           "@@M@@\\mathrm{Ery}_2@@M@@"),
    ("CD34⁺",                      "@@M@@\\mathrm{CD34}^{+}@@M@@"),
    ("1.9×10⁻⁵",         "@@M@@1.9\\times10^{-5}@@M@@"),
    ("4.3×10⁻⁷",         "@@M@@4.3\\times10^{-7}@@M@@"),
    ("10⁻⁴",                  "@@M@@10^{-4}@@M@@"),
    ("circ-R²",                    "circ-@@M@@R^2@@M@@"),
    ("linear R²",                  "linear @@M@@R^2@@M@@"),
    ("R² = −2.583",           "@@M@@R^2 = -2.583@@M@@"),
    ("R² −", "@@M@@R^2@@M@@ −"),
    ("k = 1.000", "k = 1.000"),
    ("nuclear↔surface",            "nuclear@@M@@\\leftrightarrow@@M@@surface"),
    ("GATA1↔PU.1",                 "GATA1@@M@@\\leftrightarrow@@M@@PU.1"),
    ("∮ \\mathbf{w}", "∮ \\mathbf{w}"),
]
for a, b in SPECIFIC:
    s = s.replace(a, b)
# leftover `val_corr` double-backtick guard
s = s.replace("``val_corr``", "`val_corr`")

# --- protect code spans: ASCII-ise unicode inside them, pandoc handles the rest
ASCII = {'\u00b0':' deg','\u2212':'-','\u2192':'->','\u00d7':'x','\u207b':'^-','\u1d40':'T',
         '\u221a':'sqrt','\u2081':'1','\u2082':'2','\u2083':'3','\u00b2':'2','\u03a3':'sum',
         '\u2299':'*','\u2090':'a','\u1d66':'b','\u00b7':'*','\u2016':'||'}
def _ascii_code(m):
    body = m.group(1)
    for a, b in ASCII.items():
        body = body.replace(a, b)
    return '`' + body + '`'
s = re.sub(r'`([^`]*)`', _ascii_code, s)

# ---------------------------------------------------------------- 2. single char
CHARS = {
    '−': '@@M@@-@@M@@',          # minus
    '—': '@@EMD@@',              # em dash
    '–': '@@END@@',              # en dash
    '°': '@@M@@^{\\circ}@@M@@',
    '→': '@@M@@\\rightarrow@@M@@',
    '×': '@@M@@\\times@@M@@',
    '±': '@@M@@\\pm@@M@@',
    '§': '@@SEC@@',
    '≈': '@@M@@\\approx@@M@@',
    '²': '@@M@@^{2}@@M@@',
    '⁴': '@@M@@^{4}@@M@@',
    '⁵': '@@M@@^{5}@@M@@',
    '⁷': '@@M@@^{7}@@M@@',
    '⁺': '@@M@@^{+}@@M@@',
    '⁻': '@@M@@^{-}@@M@@',
    '₁': '@@M@@_{1}@@M@@',
    '₂': '@@M@@_{2}@@M@@',
    '₃': '@@M@@_{3}@@M@@',
    'ₐ': '@@M@@_{a}@@M@@',
    'ᵦ': '@@M@@_{b}@@M@@',
    'ᵀ': '@@M@@^{\\mathsf{T}}@@M@@',
    'ρ': '@@M@@\\rho@@M@@',
    'σ': '@@M@@\\sigma@@M@@',
    'α': '@@M@@\\alpha@@M@@',
    'β': '@@M@@\\beta@@M@@',
    'θ': '@@M@@\\theta@@M@@',
    'φ': '@@M@@\\varphi@@M@@',
    'τ': '@@M@@\\tau@@M@@',
    'λ': '@@M@@\\lambda@@M@@',
    'π': '@@M@@\\pi@@M@@',
    'Δ': '@@M@@\\Delta@@M@@',
    'Σ': '@@M@@\\Sigma@@M@@',
    '·': '@@M@@\\cdot@@M@@',
    '≥': '@@M@@\\ge@@M@@',
    '≤': '@@M@@\\le@@M@@',
    '≪': '@@M@@\\ll@@M@@',
    '∈': '@@M@@\\in@@M@@',
    '⊥': '@@M@@\\perp@@M@@',
    '⊙': '@@M@@\\odot@@M@@',
    '↔': '@@M@@\\leftrightarrow@@M@@',
    '√': '@@M@@\\surd@@M@@',
    '÷': '@@M@@\\div@@M@@',
    '‖': '@@M@@\\|@@M@@',
    '✗': '@@XMARK@@',
    '…': '@@LDOTS@@',
    '̂': '',                     # combining hat (t̂)
    '∂': '@@M@@\\partial@@M@@',
    '∮': '@@M@@\\oint@@M@@',
}
for a, b in CHARS.items():
    s = s.replace(a, b)


# ---------------------------------------------------------------- 3. redaction box
s = re.sub(
    r'(?m)^> \*\*REDACTION NOTE\.\*\*(.*?)(?=\n\n)',
    lambda m: '@@REDSTART@@\n\\textbf{REDACTION NOTE.}' +
              re.sub(r'(?m)^> ?', '', m.group(1)) + '\n@@REDEND@@',
    s, flags=re.S)

# turn placeholders into pandoc raw-latex inlines so nothing gets escaped
s = re.sub(r'@@M@@(.*?)@@M@@', lambda m: '`\\(' + m.group(1) + '\\)`{=latex}', s, flags=re.S)
s = s.replace('@@EMD@@', '---').replace('@@END@@', '--')
s = s.replace('@@SEC@@', '`\\S\\,`{=latex}')
s = s.replace('@@XMARK@@', '`\\ding{55}`{=latex}')
s = s.replace('@@LDOTS@@', '...')

leftover = sorted({c for c in s if ord(c) > 127})
if leftover:
    print('WARNING unmapped:', [(c, hex(ord(c))) for c in leftover], file=sys.stderr)

open('_supp_pre.md', 'w').write(s)

# ---------------------------------------------------------------- 4. pandoc
cmd = ['pandoc', '_supp_pre.md', '-o', '_supp_raw.tex', '--to=latex', '--wrap=preserve']
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit(1)

t = open('_supp_raw.tex').read()

# ---------------------------------------------------------------- 5. restore
t = t.replace('@@REDSTART@@', '\\begin{redaction}').replace('@@REDEND@@', '\\end{redaction}')

open(OUT, 'w').write(t)
print('wrote', OUT, len(t), 'chars')
