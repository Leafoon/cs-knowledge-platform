import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";  // Add GFM support (tables, strikethrough, etc.)
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import { remarkAlerts } from "./remark-alerts";
import { rehypeInteractiveComponents } from "./rehype-interactive-components";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import rehypePrism from "rehype-prism-plus";
import { Module, TOCItem } from "@/types/content";

export function getModules(): Module[] {
    const filePath = path.join(process.cwd(), "content", "modules.json");
    const fileContents = fs.readFileSync(filePath, "utf8");
    const data = JSON.parse(fileContents);
    return data.modules;
}

export async function getModuleContent(moduleId: string) {
    const modulePath = path.join(process.cwd(), "content", moduleId);

    // Check if directory exists
    if (!fs.existsSync(modulePath)) {
        return {
            html: "<p>内容正在准备中...</p>",
            frontmatter: {},
            toc: [],
        };
    }

    const files = fs.readdirSync(modulePath)
        .filter(file => file.endsWith('.md'))
        .sort();

    if (files.length === 0) {
        return {
            html: "<p>内容正在准备中...</p>",
            frontmatter: {},
            toc: [],
        };
    }

    // Read and merge ALL markdown files
    let combinedContent = '';
    let combinedFrontmatter = {};
    const allToc: TOCItem[] = [];

    for (const markdownFile of files) {
        const fullPath = path.join(modulePath, markdownFile);
        const fileContents = fs.readFileSync(fullPath, "utf8");

        const { data, content } = matter(fileContents);

        // Merge frontmatter (first file wins for duplicates)
        if (Object.keys(combinedFrontmatter).length === 0) {
            combinedFrontmatter = data;
        }

        // Append content with separator
        if (combinedContent.length > 0) {
            combinedContent += '\n\n---\n\n';
        }
        combinedContent += content;

        // Generate TOC for this file
        const fileToc = generateTOC(content);
        allToc.push(...fileToc);
    }

    // Process combined content
    const processedContent = await unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkAlerts)
        .use(remarkRehype, { allowDangerousHtml: true })  // Convert markdown to HTML AST
        .use(rehypeInteractiveComponents)  // Transform <InteractiveComponent> tags
        .use(rehypeKatex, { strict: false })  // Disable strict mode to allow \\ in aligned/matrix environments
        .use(rehypePrism, { showLineNumbers: true, ignoreMissing: true })
        .use(rehypeStringify, { allowDangerousHtml: true })  // Stringify to HTML
        .process(combinedContent);

    const html = processedContent.toString();

    return {
        html,
        frontmatter: combinedFrontmatter,
        toc: allToc,
    };
}

export function generateTOC(markdown: string): TOCItem[] {
    // Remove code blocks to prevent headings inside code blocks from being parsed
    const cleanMarkdown = markdown.replace(/```[\s\S]*?```/g, '');

    const headingRegex = /^(#{1,3})\s+(.+)$/gm;
    const items: TOCItem[] = [];
    let match;

    // First pass: Collect all items
    while ((match = headingRegex.exec(cleanMarkdown)) !== null) {
        const level = match[1].length;
        const title = match[2].trim();
        const id = slugify(title);

        items.push({
            id,
            title,
            level,
            children: []
        });
    }

    // Second pass: Build tree structure
    const rootItems: TOCItem[] = [];
    const stack: TOCItem[] = [];

    items.forEach(item => {
        // Pop items from stack that are deeper or same level as current item
        // because they cannot be parents of current item
        while (stack.length > 0 && stack[stack.length - 1].level >= item.level) {
            stack.pop();
        }

        if (stack.length > 0) {
            // Found a parent
            const parent = stack[stack.length - 1];
            if (!parent.children) {
                parent.children = [];
            }
            parent.children.push(item);
        } else {
            // No parent, this is a root item
            rootItems.push(item);
        }

        // Push current item to stack as potential parent for next items
        stack.push(item);
    });

    return rootItems;
}

function slugify(text: string): string {
    return text
        .toString()
        .toLowerCase()
        .trim()
        .replace(/\s+/g, '-')
        .replace(/[^\w\-\u4e00-\u9fa5]+/g, '')
        .replace(/\-\-+/g, '-');
}

// 章节信息接口
export interface ChapterInfo {
    id: string;
    filename: string;
    title: string;
    description?: string;
    order: number;
}

// 获取模块的所有章节列表
export function getModuleChapters(moduleId: string): ChapterInfo[] {
    const modulePath = path.join(process.cwd(), "content", moduleId);

    if (!fs.existsSync(modulePath)) {
        return [];
    }

    // 尝试读取 chapters.json
    const chaptersJsonPath = path.join(modulePath, "chapters.json");
    if (fs.existsSync(chaptersJsonPath)) {
        const chaptersData = JSON.parse(fs.readFileSync(chaptersJsonPath, "utf8"));
        if (chaptersData.chapters && Array.isArray(chaptersData.chapters)) {
            return chaptersData.chapters
                .sort((a: any, b: any) => a.order - b.order)
                .map((chapter: any) => ({
                    id: chapter.id,
                    filename: chapter.file,
                    title: chapter.title,
                    description: chapter.description,
                    order: chapter.order
                }));
        }
    }

    // 回退到文件系统扫描（如果没有 chapters.json）
    const files = fs.readdirSync(modulePath)
        .filter(file => file.endsWith('.md') && !file.startsWith('appendix'))
        .sort();

    return files.map((filename, index) => {
        const fullPath = path.join(modulePath, filename);
        const fileContents = fs.readFileSync(fullPath, "utf8");
        const { data } = matter(fileContents);
        
        return {
            id: filename.replace('.md', ''),
            filename,
            title: data.title || filename.replace('.md', ''),
            description: data.description,
            order: index
        };
    });
}

// 获取单个章节的内容
export async function getSingleChapterContent(moduleId: string, chapterId: string) {
    const modulePath = path.join(process.cwd(), "content", moduleId);
    const chapterPath = path.join(modulePath, `${chapterId}.md`);

    if (!fs.existsSync(chapterPath)) {
        return {
            html: "<p>章节未找到</p>",
            frontmatter: {},
            toc: [],
        };
    }

    const fileContents = fs.readFileSync(chapterPath, "utf8");
    const { data, content } = matter(fileContents);

    // Process content
    const processedContent = await unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkAlerts)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeInteractiveComponents)  // Transform <InteractiveComponent> tags
        .use(rehypeKatex, { strict: false })
        .use(rehypePrism, { showLineNumbers: true, ignoreMissing: true })
        .use(rehypeStringify, { allowDangerousHtml: true })
        .process(content);

    const html = processedContent.toString();
    const toc = generateTOC(content);

    return {
        html,
        frontmatter: data,
        toc,
    };
}
