---
name: interactive-components
description: Create and register React interactive components for the CS Knowledge Platform - batch creation, registration, and build verification
---

# Interactive Components Skill

Create and register React interactive components for the CS Knowledge Platform following established patterns and quality standards.

## When to Use

- User requests creation of interactive components for a module
- User reports "[Error: Component "X" not found]" errors
- User requests batch creation of multiple components

## Workflow

### Phase 1: Check Existing Components
1. Check `components/interactive/` directory for existing `.tsx` files
2. Check `components/interactive/index.ts` for barrel exports
3. Check `components/knowledge/ContentRenderer.tsx` for component mapping
4. Identify missing components referenced in markdown files

### Phase 2: Create Components
1. Create `.tsx` files in `components/interactive/`
2. Follow component format conventions (see below)
3. Use batch creation: 5-10 components per subagent
4. Avoid duplicate file targets across subagents

### Phase 3: Register Components
1. Add exports to `components/interactive/index.ts`
2. Add mapping to `components/knowledge/ContentRenderer.tsx`
3. Ensure component names match between markdown tags and registration

### Phase 4: Verify Build
1. Run `npm run build`
2. Check for TypeScript/JSX errors
3. Fix common issues (see known patterns)
4. Verify components render correctly

## Component Format Template

```tsx
"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Info, Play, RotateCcw } from "lucide-react";

interface DataType {
  id: number;
  label: string;
  value: string;
}

const DATA: DataType[] = [
  { id: 1, label: "Item 1", value: "Value 1" },
  { id: 2, label: "Item 2", value: "Value 2" },
];

export function ComponentName() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4">Component Title</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Interactive content */}
      </div>
    </div>
  );
}
```

## Registration Pattern

### index.ts (barrel export):
```typescript
export { ComponentName } from "./ComponentName";
```

### ContentRenderer.tsx (mapping):
```typescript
import { ComponentName } from "../interactive";

const componentMap = {
  ComponentName: ComponentName,
  // ... other components
};
```

## Known Patterns & Pitfalls

### From MEMORY.md:
- Use `"use client"` at top of every component
- Function names must start with uppercase (react-hooks/rules-of-hooks)
- Use `useState` for interactivity, `framer-motion` for animations
- Dark mode support with `dark:` variants
- Standard outer container with `max-w-6xl`

### JSX Text Escaping:
- `->` must be `{'->'}` or `{"->"}`
- `//` must be `{'//'}` or `{"// "}`
- Standalone `>` must be `{">"}`
- Object keys with hyphens must be quoted: `"C-SCAN": value`

### TypeScript Issues:
- Use `Number(x.toFixed(1))` instead of `x.toFixed(1) as unknown as number`
- Avoid complex generics in `useRef<ReturnType<typeof setInterval> | null>`
- Use `useState<string[]>([])` not `useState([] as string[])`

### Build Errors:
- ESLint errors may exist even with "✓ Compiled successfully"
- Must grep build output for all `Error:` lines
- Clear cache: `rm -rf .next && npm run build`

### Subagent Management:
- Split large batches (200+ components) into 6 parallel agents
- Each agent writes ~35 `.tsx` files
- Instruct subagents to ONLY create files, not touch registration files
- Main agent handles registration to avoid duplicates

## Component Quality Checklist

- [ ] Has `"use client"` directive
- [ ] Uses `useState` for interactivity
- [ ] Uses `framer-motion` for animations
- [ ] Uses `lucide-react` for icons
- [ ] Has dark mode support with `dark:` variants
- [ ] Uses `max-w-6xl` outer container
- [ ] Exported as named export
- [ ] Function name starts with uppercase
- [ ] No JSX text escaping errors
- [ ] No TypeScript type errors
- [ ] Registered in `index.ts` and `ContentRenderer.tsx`
- [ ] Build succeeds without errors

## Common Component Types

### Comparison Tables
- Side-by-side feature comparisons
- Interactive row selection
- Highlighting differences

### Flow Diagrams
- Step-by-step process visualization
- Animated transitions between states
- Interactive navigation

### Data Structure Visualizations
- Tree/graph animations
- Array/list operations
- State machine diagrams

### Interactive Examples
- Code execution simulations
- Parameter tuning interfaces
- Before/after comparisons

### Performance Benchmarks
- Timing comparisons
- Resource usage charts
- Scalability demonstrations