import re
import os

index_path = 'components/interactive/index.ts'
cr_path = 'components/knowledge/ContentRenderer.tsx'

# Step 1: Get all valid exports from index.ts
with open(index_path) as f:
    idx_content = f.read()

# Match all export names: "export { X }" or "export { X as Y }" or "export default as X"
valid_exports = set()
for m in re.finditer(r'export\s+\{\s*(?:default\s+as\s+)?(\w+)', idx_content):
    valid_exports.add(m.group(1))

# Also match "export { default as X }"
for m in re.finditer(r'export\s+\{\s*default\s+as\s+(\w+)', idx_content):
    valid_exports.add(m.group(1))

print(f"Found {len(valid_exports)} valid exports in index.ts")

# Step 2: Read ContentRenderer.tsx
with open(cr_path) as f:
    cr_lines = f.readlines()

# Step 3: Find the import block from @/components/interactive
import_start = None
import_end = None
for i, line in enumerate(cr_lines):
    if 'from "@/components/interactive"' in line:
        import_end = i
        # Find the start of the import statement
        for j in range(i, -1, -1):
            if re.match(r'^import\s*\{', cr_lines[j]) or re.match(r'^import\s+\{', cr_lines[j]):
                import_start = j
                break
        break

if import_start is None:
    print("ERROR: Could not find import block")
    exit(1)

print(f"Import block: lines {import_start+1}-{import_end+1}")

# Step 4: Extract all imported names from the import block
import_text = ''.join(cr_lines[import_start:import_end+1])
imported_names = set()
for m in re.finditer(r'(\w+)(?:\s+as\s+\w+)?', import_text):
    name = m.group(1)
    # Skip keywords and 'import', 'from', 'as'
    if name in ('import', 'from', 'as'):
        continue
    imported_names.add(name)

# Filter: only keep names that exist as .tsx files in components/interactive/
valid_imported = set()
invalid_imported = set()
for name in imported_names:
    # Check for exact file
    if os.path.exists(f'components/interactive/{name}.tsx') or os.path.exists(f'components/interactive/{name}.ts'):
        valid_imported.add(name)
    else:
        invalid_imported.add(name)

print(f"Valid imports: {len(valid_imported)}")
print(f"Invalid imports: {len(invalid_imported)}")
if invalid_imported:
    print(f"Removing: {sorted(invalid_imported)[:10]}...")

# Step 5: Rebuild the import block
# Parse the original import into individual names with their line positions
new_import_lines = []
for i in range(import_start, import_end + 1):
    line = cr_lines[i]
    # Extract names from this line
    names_in_line = []
    for m in re.finditer(r'(\w+)(?:\s+as\s+(\w+))?(?=\s*[,}\n])', line):
        orig_name = m.group(1)
        alias = m.group(2)
        if orig_name in ('import', 'from', 'as'):
            continue
        if orig_name in valid_imported:
            names_in_line.append((orig_name, alias))
        elif orig_name in invalid_imported:
            pass  # skip invalid
        elif orig_name not in ('import', 'from', 'as', ''):
            # Might be from another module, keep it
            names_in_line.append((orig_name, alias))
    
    if names_in_line:
        # Rebuild this line with only valid names
        text = line
        for orig_name, alias in names_in_line:
            # This approach is too fragile, let's use a different approach
            pass
    
    # Simpler approach: just mark lines to keep/remove
    # We'll rebuild the entire import block

# Step 5b: Rebuild import block from scratch
# Collect all component names that should be imported (valid imports)
# Group them by original order
ordered_names = []
seen = set()
for i in range(import_start, import_end + 1):
    line = cr_lines[i]
    for m in re.finditer(r'(\w+)(?:\s+as\s+(\w+))?(?=\s*[,}\n])', line):
        orig_name = m.group(1)
        alias = m.group(2)
        if orig_name in ('import', 'from', 'as', ''):
            continue
        if orig_name in seen:
            continue
        seen.add(orig_name)
        if orig_name in valid_imported:
            ordered_names.append((orig_name, alias))
        elif orig_name not in invalid_imported:
            # Not in our list at all, might be from another import
            ordered_names.append((orig_name, alias))

# Group into lines of ~5 names each
new_lines = []
chunk_size = 5
for i in range(0, len(ordered_names), chunk_size):
    chunk = ordered_names[i:i+chunk_size]
    parts = []
    for orig_name, alias in chunk:
        if alias:
            parts.append(f"{orig_name} as {alias}")
        else:
            parts.append(orig_name)
    line = "    " + ", ".join(parts) + ",\n"
    new_lines.append(line)

# Build the final import statement
if new_lines:
    # Make last line not have trailing comma before }
    new_lines[-1] = new_lines[-1].rstrip().rstrip(',') + '\n'
    new_import = "import {\n" + "".join(new_lines) + '} from "@/components/interactive";\n'
else:
    new_import = 'import {} from "@/components/interactive";\n'

# Replace the import block
new_cr_lines = cr_lines[:import_start] + [new_import] + cr_lines[import_end+1:]

# Step 6: Now clean up the component map
# Find all componentMap entries and remove references to invalid components
# We need to find the componentMap and clean it up
# Read the full new content
new_content = ''.join(new_cr_lines)

# Find all component names used in the component map that are not valid
# The component map typically looks like: ComponentName: ComponentName,
# or "ComponentName": ComponentName,
map_pattern = re.compile(r'^\s*(?:(\w+)|"(\w+)")\s*:\s*(\w+)\s*,?\s*$', re.MULTILINE)

invalid_in_map = []
for m in map_pattern.finditer(new_content):
    key = m.group(1) or m.group(2)
    value = m.group(3)
    if value not in valid_imported and value in invalid_imported:
        invalid_in_map.append((m.start(), m.end(), m.group(0)))

# Remove invalid map entries (in reverse order to preserve positions)
for start, end, text in reversed(invalid_in_map):
    new_content = new_content[:start] + new_content[end:]

# Also handle Object.assign blocks that might reference missing components
# Find lines with "MissingComponentName," in map sections
# Remove individual lines from the import and map

with open(cr_path, 'w') as f:
    f.write(new_content)

print(f"\nFinal: {len(valid_imported)} components imported")
print(f"Removed {len(invalid_in_map)} invalid component map entries")
