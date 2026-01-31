import { getModuleContent, getModules, getModuleChapters } from "@/lib/content-loader";
import { ContentRenderer } from "@/components/knowledge/ContentRenderer";
import { Badge } from "@/components/ui/Badge";
import Link from "next/link";

export async function generateStaticParams() {
    const modules = getModules();
    return modules.map((module) => ({
        module: module.id,
    }));
}

export default async function ModulePage({
    params,
}: {
    params: { module: string };
}) {
    const modules = getModules();
    const currentModule = modules.find((m) => m.id === params.module);

    // 检查是否有章节（多页模式）
    const chapters = getModuleChapters(params.module);

    // 如果有章节，显示章节目录
    if (chapters.length > 0) {
        return (
            <div className="min-h-screen bg-slate-50 dark:bg-black font-sans selection:bg-indigo-500/30">
                {/* 1. Immersive Dark Header Section */}
                <div className="relative bg-[#0F172A] pt-32 pb-48 px-6 sm:px-12 overflow-hidden border-b border-slate-800">
                    {/* Technical Grid Pattern */}
                    <div className="absolute inset-0 opacity-[0.08]"
                        style={{ backgroundImage: 'linear-gradient(#94a3b8 1px, transparent 1px), linear-gradient(to right, #94a3b8 1px, transparent 1px)', backgroundSize: '40px 40px' }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-[#0F172A] via-transparent to-transparent" />

                    <div className="relative max-w-7xl mx-auto z-10">
                        <div className="flex flex-col lg:flex-row gap-12 items-start lg:items-end justify-between">
                            <div className="max-w-3xl">
                                {/* Breadcrumb / Badge */}
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-mono font-medium uppercase tracking-wider mb-8">
                                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
                                    Module / {params.module}
                                </div>

                                <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white mb-6 leading-tight">
                                    {currentModule?.title || params.module}
                                </h1>

                                <p className="text-lg md:text-xl text-slate-400 max-w-2xl leading-relaxed font-light">
                                    {currentModule?.description}
                                </p>
                            </div>

                            {/* Course Stats Card */}
                            <div className="flex items-center gap-8 bg-slate-800/50 backdrop-blur-md border border-slate-700/50 p-6 rounded-2xl min-w-[280px]">
                                <div>
                                    <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1">Chapters</div>
                                    <div className="text-3xl font-mono text-white tracking-tighter">{chapters.length.toString().padStart(2, '0')}</div>
                                </div>
                                <div className="w-px h-10 bg-slate-700" />
                                <div>
                                    <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1">Time</div>
                                    <div className="text-white font-medium">~ 3 Hours</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* 2. Overlapping Card Grid */}
                <div className="relative max-w-7xl mx-auto px-6 sm:px-12 -mt-24 pb-24 z-20">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {chapters.map((chapter, index) => (
                            <Link
                                key={chapter.id}
                                href={`/${params.module}/${chapter.id}`}
                                className="group relative flex flex-col h-full bg-white dark:bg-slate-900 rounded-2xl p-8 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)] hover:shadow-[0_20px_40px_-15px_rgba(6,81,237,0.2)] dark:shadow-none dark:hover:shadow-slate-800/50 border border-slate-100 dark:border-slate-800 transition-all duration-300 hover:-translate-y-1"
                            >
                                {/* Card Header: Number & Status */}
                                <div className="flex justify-between items-start mb-8">
                                    <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-lg font-bold border border-slate-100 dark:border-slate-700 group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-500 transition-colors duration-300">
                                        {index + 1}
                                    </div>
                                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 transform translate-x-2 group-hover:translate-x-0">
                                        <svg className="w-6 h-6 text-indigo-600 dark:text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                        </svg>
                                    </div>
                                </div>

                                {/* Card Content */}
                                <div className="flex-grow">
                                    <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                                        {chapter.title.replace(/^第\s*(\d+)\s*章\s*[：:]?\s*/, "Chapter $1: ").replace(/^Chapter\s*(\d+)\s*[\.:]?\s*/i, "Chapter $1: ")}
                                    </h3>
                                    {chapter.description && (
                                        <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed line-clamp-3">
                                            {chapter.description}
                                        </p>
                                    )}
                                </div>

                                {/* Card Footer: Progress Bar (Decorative) */}
                                <div className="mt-8">
                                    <div className="w-full h-1 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full w-0 bg-indigo-500 group-hover:w-full transition-all duration-700 ease-out" />
                                    </div>
                                    <div className="flex justify-between mt-2 text-[10px] uppercase font-bold tracking-widest text-slate-400">
                                        <span>Status</span>
                                        <span className="group-hover:text-indigo-600 transition-colors">Start</span>
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    // 如果没有章节，使用原来的单页模式
    const { html, frontmatter } = await getModuleContent(params.module);

    return (
        <article className="min-h-screen">
            {/* Module Header */}
            <header className="relative mb-16 pb-12 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/5 to-transparent -z-10" />

                <div className="relative">
                    <div className="flex flex-wrap items-center gap-3 mb-8">
                        <Badge variant="default" className="bg-gradient-to-r from-accent-primary/15 to-accent-secondary/15 text-accent-primary border-accent-primary/30 font-semibold px-4 py-1.5 shadow-sm">
                            <svg className="w-3.5 h-3.5 inline mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
                            </svg>
                            知识模块
                        </Badge>
                        {(frontmatter as any).updated && (
                            <div className="flex items-center gap-2 text-sm text-text-tertiary bg-bg-elevated/50 px-3 py-1.5 rounded-full backdrop-blur-sm border border-border-subtle/50">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span className="font-medium">最近更新 {(frontmatter as any).updated}</span>
                            </div>
                        )}
                    </div>

                    <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-br from-text-primary via-accent-primary to-accent-secondary mb-6 leading-tight tracking-tight">
                        {currentModule?.title || params.module}
                    </h1>

                    <p className="text-lg md:text-xl text-text-secondary max-w-3xl leading-relaxed font-light">
                        {currentModule?.description || (frontmatter as any).description}
                    </p>

                    <div className="mt-10 h-1 w-24 bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full" />
                </div>
            </header>

            {/* Content with embedded interactive components */}
            <ContentRenderer html={html} moduleId={params.module} />
        </article>
    );
}
