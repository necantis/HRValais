"""Fix non-Latin1 chars in pdf_generator.py for Helvetica compatibility."""
path = 'utils/pdf_generator.py'
content = open(path, encoding='utf-8').read()
# Replace em-dash (U+2014) with ASCII hyphen
content = content.replace('\u2014', ' - ')
# Replace copyright symbol (U+00A9) — also not in standard Helvetica subset
content = content.replace('\u00a9', '(c)')
open(path, 'w', encoding='utf-8').write(content)

# Also delete the existing (possibly cached) PDF so it regenerates
import os
pdf_path = 'assets/fiches_pratiques.pdf'
if os.path.exists(pdf_path):
    os.remove(pdf_path)
    print(f"Deleted {pdf_path} to force PDF regeneration")
print("pdf_generator.py patched for Latin-1 Helvetica font")
