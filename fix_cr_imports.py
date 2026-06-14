import re
import os

cr_path = 'components/knowledge/ContentRenderer.tsx'

# Get list of all existing component files
existing = set()
for f in os.listdir('components/interactive'):
    if f.endswith('.tsx') or f.endswith('.ts'):
        existing.add(f.rsplit('.', 1)[0])

print(f"Found {len(existing)} existing component files")

with open(cr_path) as f:
    content = f.read()

# Step 1: Fix import block (lines 5-189)
# Find the import statement
import_match = re.search(r'import\s*\{([^}]+)\}\s*from\s*["\']@/components/interactive["\'];', content, re.DOTALL)
if not import_match:
    print("ERROR: import block not found")
    exit(1)

old_import = import_match.group(0)
names_block = import_match.group(1)

# Parse names
all_names = []
for m in re.finditer(r'(\w+)(?:\s+as\s+(\w+))?', names_block):
    orig, alias = m.group(1), m.group(2)
    if orig not in ('import', 'from', 'as'):
        all_names.append((orig, alias))

# Filter
keep = [(o, a) for o, a in all_names if o in existing]
remove = [o for o, a in all_names if o not in existing]
print(f"Import: keeping {len(keep)}, removing {len(remove)}")

# Rebuild import
chunks = []
for i in range(0, len(keep), 5):
    parts = []
    for o, a in keep[i:i+5]:
        parts.append(f"{o} as {a}" if a else o)
    chunks.append("    " + ", ".join(parts) + ",")
if chunks:
    chunks[-1] = chunks[-1].rstrip().rstrip(',')
new_import = "import {\n" + "\n".join(chunks) + '\n} from "@/components/interactive";'
content = content[:import_match.start()] + new_import + content[import_match.end():]

# Step 2: Remove invalid componentMap entries
# Find all lines like "Name": Name, or Name: Name, that reference removed components
lines = content.split('\n')
new_lines = []
in_map = False
brace_depth = 0
for line in lines:
    stripped = line.strip()
    
    # Track if we're in the componentMap
    if 'const componentMap' in line and '=' in line:
        in_map = True
        brace_depth = 0
    
    if in_map:
        # Count braces
        brace_depth += line.count('{') - line.count('}')
        if brace_depth <= 0 and '}' in line:
            in_map = False
        
        # Check if this line maps a removed component
        m = re.match(r'^\s*(?:"(\w+)"|(\w+))\s*:\s*(\w+)\s*,?\s*$', stripped)
        if m:
            value = m.group(3)
            if value in remove:
                continue  # skip this line
    
    new_lines.append(line)

content = '\n'.join(new_lines)

# Step 3: Also remove Object.assign entries for missing components
# Pattern: Object.assign(componentMap, { ... }) blocks
assign_pattern = re.compile(r'Object\.assign\(\w+,\s*\{([^}]+)\}\)', re.DOTALL)
for m in assign_pattern.finditer(content):
    block = m.group(1)
    # Check if any entries reference removed components
    has_invalid = False
    for name in remove:
        if name in block:
            has_invalid = True
            break
    if has_invalid:
        # Remove entire Object.assign block
        content = content[:m.start()] + content[m.end():]

# Clean up multiple blank lines
content = re.sub(r'\n{4,}', '\n\n\n', content)

with open(cr_path, 'w') as f:
    f.write(content)

print(f"Done! Cleaned up import and componentMap.")
