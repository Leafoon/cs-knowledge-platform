import { getSingleChapterContent, getModuleChapters, getModules } from "@/lib/content-loader";
import { ContentRenderer } from "@/components/knowledge/ContentRenderer";
import { TOC } from "@/components/knowledge/TOC";
import { OnPageTOC } from "@/components/knowledge/OnPageTOC";
import { Badge } from "@/components/ui/Badge";
import Link from "next/link";
import { notFound } from "next/navigation";

export async function generateStaticParams() {
    const modules = getModules();
    const params: { module: string; chapter: string }[] = [];
    
    for (const module of modules) {
        const chapters = getModuleChapters(module.id);
        for (const chapter of chapters) {
            params.push({
                module: module.id,
                chapter: chapter.id,
            });
        }
    }
    
    return params;
}

export default async function ChapterPage({
    params,
}: {
    params: { module: string; chapter: string };
}) {
    const { html, frontmatter, toc } = await getSingleChapterContent(params.module, params.chapter);
    const modules = getModules();
    const currentModuleData = modules.find((m) => m.id === params.module);
    const chapters = getModuleChapters(params.module);
    const currentChapterIndex = chapters.findIndex((ch) => ch.id === params.chapter);
    
    if (currentChapterIndex === -1) {
        notFound();
    }
    
    const currentChapter = chapters[currentChapterIndex];
    const prevChapter = currentChapterIndex > 0 ? chapters[currentChapterIndex - 1] : null;
    const nextChapter = currentChapterIndex < chapters.length - 1 ? chapters[currentChapterIndex + 1] : null;

    return (
        <div className="flex gap-8 max-w-[1600px] mx-auto px-6">
            {/* Left Sidebar - Chapter TOC */}
            {toc.length > 0 && (
                <aside className="hidden xl:block w-64 flex-shrink-0">
                    <div className="sticky top-20">
                        <div className="text-xs font-semibold text-text-tertiary uppercase tracking-wide mb-4">
                            本章目录
                        </div>
                        <TOC items={toc} activeId="" />
                    </div>
                </aside>
            )}

            {/* Main Content */}
            <article className="flex-1 min-w-0 pb-16">
                {/* Breadcrumb Navigation */}
                <nav className="mb-8 flex items-center gap-2 text-sm text-text-tertiary">
                    <Link href="/" className="hover:text-text-primary transition-colors">
                        首页
                    </Link>
                    <span>/</span>
                    <Link href={`/${params.module}`} className="hover:text-text-primary transition-colors">
                        {currentModuleData?.title}
                    </Link>
                    <span>/</span>
                    <span className="text-text-secondary">{currentChapter.title}</span>
                </nav>

                {/* Chapter Header */}
                <header className="mb-16 pb-8 border-b border-border-subtle">
                    {/* Meta Info */}
                    <div className="flex items-center gap-6 mb-6 text-sm text-text-tertiary">
                        <div className="flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                            </svg>
                            <span className="font-medium">Chapter {currentChapterIndex + 1} of {chapters.length}</span>
                        </div>
                        {(frontmatter as any).updated && (
                            <>
                                <span>·</span>
                                <div className="flex items-center gap-2">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <span>更新于 {(frontmatter as any).updated}</span>
                                </div>
                            </>
                        )}
                    </div>

                    {/* Title */}
                    <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-4 leading-tight tracking-tight">
                        {(frontmatter as any).title || currentChapter.title}
                    </h1>

                    {/* Description */}
                    {(frontmatter as any).description && (
                        <p className="text-lg text-text-tertiary leading-relaxed max-w-3xl">
                            {(frontmatter as any).description}
                        </p>
                    )}
                </header>

                {/* Content */}
                <div className="max-w-3xl">
                    <ContentRenderer html={html} moduleId={params.module} />
                </div>

                {/* Chapter Navigation */}
                <nav className="mt-20 pt-8 border-t border-border-subtle max-w-3xl">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Previous */}
                        {prevChapter ? (
                            <Link
                                href={`/${params.module}/${prevChapter.id}`}
                                className="group flex items-center gap-3 p-4 rounded-lg border border-border-subtle hover:border-accent-primary/50 hover:bg-bg-elevated transition-all"
                            >
                                <svg className="w-5 h-5 text-text-tertiary group-hover:text-accent-primary group-hover:-translate-x-1 transition-all flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                </svg>
                                <div className="flex-1 min-w-0 text-left">
                                    <div className="text-xs text-text-tertiary mb-1">上一章</div>
                                    <div className="text-sm font-medium text-text-primary truncate">{prevChapter.title}</div>
                                </div>
                            </Link>
                        ) : (
                            <div />
                        )}

                        {/* Index */}
                        <Link
                            href={`/${params.module}`}
                            className="flex items-center justify-center gap-2 p-4 rounded-lg bg-bg-elevated hover:bg-accent-primary/10 border border-border-subtle hover:border-accent-primary/50 transition-all"
                        >
                            <svg className="w-4 h-4 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                            <span className="text-sm font-medium text-text-primary">目录</span>
                        </Link>

                        {/* Next */}
                        {nextChapter ? (
                            <Link
                                href={`/${params.module}/${nextChapter.id}`}
                                className="group flex items-center gap-3 p-4 rounded-lg border border-border-subtle hover:border-accent-primary/50 hover:bg-bg-elevated transition-all"
                            >
                                <div className="flex-1 min-w-0 text-right">
                                    <div className="text-xs text-text-tertiary mb-1">下一章</div>
                                    <div className="text-sm font-medium text-text-primary truncate">{nextChapter.title}</div>
                                </div>
                                <svg className="w-5 h-5 text-text-tertiary group-hover:text-accent-primary group-hover:translate-x-1 transition-all flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </Link>
                        ) : (
                            <div />
                        )}
                    </div>

                    {/* Progress */}
                    <div className="mt-6 flex items-center justify-center gap-3 text-sm text-text-tertiary">
                        <div className="w-48 h-1.5 bg-border-subtle rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-accent-primary rounded-full transition-all"
                                style={{ width: `${((currentChapterIndex + 1) / chapters.length) * 100}%` }}
                            />
                        </div>
                        <span className="font-medium tabular-nums">
                            {currentChapterIndex + 1} / {chapters.length}
                        </span>
                    </div>
                </nav>
            </article>

            {/* Right Sidebar - On-Page TOC */}
            <aside className="hidden xl:block w-64 flex-shrink-0">
                <div className="sticky top-20">
                    <OnPageTOC />
                </div>
            </aside>
        </div>
    );
}
