import os

content = open('E:\\projects\\BagBuddy\\config.py', 'r', encoding='utf-8').read()

old = '# Langfuse (from .env.public)\nLANGFUSE_HOST'
new = '# Langfuse keys (from .env, gitignored - never committed)\nLANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")\nLANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")\nLANGFUSE_HOST'

content = content.replace(old, new)
open('E:\\projects\\BagBuddy\\config.py', 'w', encoding='utf-8').write(content)
print('config.py updated')
