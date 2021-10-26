import re

with open('../../setup.py') as f:
    contents = f.read()

# Find install_requires
match = re.search(r'install_requires=\[.*?],', contents, re.MULTILINE | re.DOTALL)
# Get the individual lines, throw away cruft
opening_bracket, *reqs, closing_bracket = match.group().splitlines()
# Remove the quotes and commas
reqs = [r.strip().strip('\',') for r in reqs]
# Write the result
reqs.insert(0, '# Auto-generated from setup.py with generate-requirements.in.py')
with open('requirements.in', 'w') as f:
    f.write('\n'.join(reqs))
