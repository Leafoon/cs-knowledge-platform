---
name: content-generation
description: Generate educational content for the CS Knowledge Platform - create module outlines, chapter markdown files, and expand content to meet line count requirements
---

# Content Generation Skill

Generate educational content for the CS Knowledge Platform following established patterns and quality standards.

## When to Use

- User requests generation of content for a new module (e.g., "生成 TileLang 教学内容", "为 Triton 生成 30+ 章节")
- User requests expansion of existing chapters to meet line count requirements
- User requests batch generation of multiple chapters

## Workflow

### Phase 1: Check Existing Content
1. Check if module directory exists: `content/<module>/`
2. Check if `chapters.json` exists
3. Check existing chapter files and their line counts
4. Check `content/modules.json` for module entry

### Phase 2: Generate Outline
1. Create detailed outline with 30+ chapters
2. Follow format: Chapter 0, Chapter 1, etc.
3. Include 2-3 level hierarchy (chapter → section → subsection)
4. Ensure logical progression from basics to advanced topics

### Phase 3: Generate Content
1. Generate chapters one by one or in batches
2. Follow chapter format conventions:
   - `# Chapter N: Title` heading
   - `> **学习目标**：` block with 5-6 bullets
   - `## N.N` section numbering
   - Code blocks with language tags
   - `<div data-component="X"></div>` for interactive components
   - Comparison tables
   - `## 本章小结` and `## 思考题`
3. Minimum 1000 lines per chapter (2000+ for advanced modules)
4. Chinese explanations, English code

### Phase 4: Expand Content
1. Check line counts: `wc -l content/<module>/*.md`
2. Expand chapters below minimum using `edit` tool
3. Add: deeper explanations, more code examples, comparison tables, case studies, performance benchmarks, best practices

### Phase 5: Verify Build
1. Run `npm run build`
2. Check for errors: `grep -E "Error:|Failed" build.log`
3. Fix JSX/TSX errors (see known patterns below)
4. Clear cache if needed: `rm -rf .next && npm run build`

## Chapter Format Template

```markdown
---
title: "Chapter N: Title"
description: "Brief description"
updated: "YYYY-MM-DD"
---

# Chapter N: Title

> **学习目标**：
> - Objective 1
> - Objective 2
> - Objective 3
> - Objective 4
> - Objective 5

## N.1 Section Title

Content with code blocks:

```python
# Code example with Chinese comments
def example():
    pass
```

<div data-component="ComponentName"></div>

## N.2 Another Section

Content with comparison table:

| Feature | Option A | Option B |
|---------|----------|----------|
| ... | ... | ... |

## 本章小结

Summary of key points.

## 思考题

1. Question 1
2. Question 2
3. Question 3
```

## Known Patterns & Pitfalls

### From MEMORY.md:
- Each chapter must have **at least 1000 lines** (except index/overview pages)
- Content must be **teaching-style** (教学式), detailed, rich, in-depth
- Do **not** use code to pad line counts
- Generate detailed outline first before writing chapter content
- Interactive components must use `<div data-component="X"></div>` format (NOT `[组件：X]`)

### Build Errors:
- JSX text escaping: `->` must be `{'->'}`, `//` must be `{'//'}`, `>` must be `{">"}`
- Component naming: Function names must start with uppercase
- Object keys with hyphens must be quoted: `"C-SCAN": value`
- SWC parser sensitivity: Simplify TypeScript patterns

### Subagent Management:
- One chapter per subagent for 2000+ line content
- 3 chapters per subagent max for heavy content
- Check file existence rather than relying on agent status
- Avoid duplicate file targets across subagents

## Quality Checklist

- [ ] Chapter has ≥1000 lines (2000+ for advanced modules)
- [ ] Uses `## N.N` section numbering
- [ ] Includes `> **学习目标**：` block
- [ ] Has `<div data-component="X"></div>` tags (not bracket format)
- [ ] Includes code blocks with language tags
- [ ] Contains comparison tables where appropriate
- [ ] Ends with `## 本章小结` and `## 思考题`
- [ ] Chinese explanations, English code
- [ ] No JSX text escaping errors
- [ ] Build succeeds without errors