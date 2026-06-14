---
name: build-verification
description: Verify builds, fix JSX/TSX errors, and troubleshoot common build issues in the CS Knowledge Platform
---

# Build Verification Skill

Verify builds, fix JSX/TSX errors, and troubleshoot common build issues in the CS Knowledge Platform.

## When to Use

- After generating or modifying content/components
- When encountering build errors
- When user reports rendering issues

## Workflow

### Phase 1: Run Build
1. Run `npm run build`
2. Capture output to file: `npm run build 2>&1 | tee build.log`
3. Check for success: `grep -E "Compiled successfully|✓" build.log`

### Phase 2: Check for Errors
1. Search for errors: `grep -E "Error:|Failed|error" build.log`
2. Categorize errors (JSX, TypeScript, ESLint, etc.)
3. Identify affected files

### Phase 3: Fix Errors
1. Fix JSX text escaping issues
2. Fix TypeScript type errors
3. Fix ESLint warnings
4. Fix component naming issues
5. Clear cache if needed

### Phase 4: Verify Fix
1. Re-run build
2. Check for new errors
3. Verify component rendering

## Common Error Patterns

### JSX Text Escaping
**Error**: `Expected a semicolon` or `',' expected`
**Fix**:
- `->` → `{'->'}` or `{"->"}`
- `//` → `{'//'}` or `{"// "}`
- `>` → `{">"}`
- Object keys with hyphens: `"C-SCAN": value`

### TypeScript Type Errors
**Error**: `Type 'string' is not assignable to type 'number'`
**Fix**:
- Use `Number(x.toFixed(1))` instead of `x.toFixed(1) as unknown as number`
- Use `useState<string[]>([])` not `useState([] as string[])`
- Avoid complex generics in `useRef`

### Component Naming
**Error**: `react-hooks/rules-of-hooks`
**Fix**:
- Function names must start with uppercase
- Use `export { UpperName as lowerName }` for backward compatibility

### ESLint Errors
**Error**: `ESLintError: ...`
**Fix**:
- May exist even with "✓ Compiled successfully"
- Must grep build output for all `Error:` lines
- Fix each error individually

### Cache Issues
**Error**: Stale cache causing false errors
**Fix**:
- `rm -rf .next && npm run build`
- Clear `.next/` directory
- Restart dev server

## Build Commands

### Basic Build
```bash
npm run build
```

### Build with Error Capture
```bash
npm run build 2>&1 | tee build.log
```

### Check Build Output
```bash
grep -E "Error:|Failed|error" build.log
grep -E "Compiled successfully|✓" build.log
```

### Clear Cache
```bash
rm -rf .next && npm run build
```

### TypeScript Check
```bash
npx tsc --noEmit 2>&1
```

## Error Troubleshooting Guide

### JSX Text Escaping Errors
1. Find affected file: `grep -rn "->" components/`
2. Replace with escaped version
3. Check for `//` and `>` as well
4. Rebuild and verify

### TypeScript Errors
1. Find affected file: `npx tsc --noEmit 2>&1 | grep "error TS"`
2. Check type annotations
3. Simplify complex types
4. Rebuild and verify

### Component Not Found Errors
1. Check component exists in `components/interactive/`
2. Check export in `index.ts`
3. Check mapping in `ContentRenderer.tsx`
4. Check markdown tag format: `<div data-component="X"></div>`

### Build Stale Cache
1. Delete `.next/` directory
2. Rebuild: `npm run build`
3. If still fails, restart dev server

## Verification Checklist

- [ ] Build completes without errors
- [ ] No ESLint errors in output
- [ ] No TypeScript errors
- [ ] All components render correctly
- [ ] No "Component not found" errors
- [ ] Markdown content displays properly
- [ ] Interactive components work

## Known Patterns from MEMORY.md

- `npm run build` may show "✓ Compiled successfully" but still fail if ESLint errors exist
- Must grep build output for all `Error:` lines
- SWC parser sensitivity: Simplify TypeScript patterns
- Duplicate variable declaration in subagent output
- StepLog/interface required field pattern
- TypeScript type error iteration: Fix all entries at once