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
            <div className="min-h-screen pb-20">
                {/* Decorative Background Elements */}
                <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
                    <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent-primary/5 rounded-full blur-3xl" />
                    <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent-secondary/5 rounded-full blur-3xl" />
                </div>

                {/* Module Header */}
                <header className="relative mb-24 pb-20 overflow-hidden">
                    {/* Background Pattern */}
                    <div className="absolute inset-0 -z-10">
                        <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/8 via-accent-primary/3 to-transparent" />
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(var(--accent-primary-rgb),0.1),transparent_50%)]" />
                    </div>

                    <div className="relative max-w-7xl mx-auto px-6">
                        {/* Top Badges */}
                        <div className="flex flex-wrap items-center gap-4 mb-12 animate-fade-in">
                            <Badge 
                                variant="default" 
                                className="bg-gradient-to-r from-accent-primary/20 to-accent-secondary/20 text-accent-primary border-accent-primary/40 font-bold px-5 py-2.5 shadow-lg backdrop-blur-sm"
                            >
                                <svg className="w-5 h-5 inline mr-2.5" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
                                </svg>
                                <span className="text-sm">知识模块</span>
                            </Badge>
                            <div className="flex items-center gap-3 text-sm bg-bg-elevated/80 px-5 py-2.5 rounded-full backdrop-blur-md border border-border-subtle/60 shadow-md">
                                <svg className="w-5 h-5 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                </svg>
                                <span className="font-bold text-text-primary">{chapters.length}</span>
                                <span className="text-text-tertiary font-medium">章节课程</span>
                            </div>
                        </div>

                        {/* Main Title */}
                        <div className="mb-10">
                            <h1 className="text-6xl md:text-7xl lg:text-8xl font-black text-transparent bg-clip-text bg-gradient-to-br from-text-primary via-accent-primary to-accent-secondary mb-10 leading-[1.1] tracking-tight drop-shadow-sm">
                                {currentModule?.title || params.module}
                            </h1>
                            
                            <p className="text-xl md:text-2xl lg:text-3xl text-text-secondary max-w-4xl leading-relaxed font-light">
                                {currentModule?.description}
                            </p>
                        </div>
                        
                        {/* Decorative Line */}
                        <div className="flex items-center gap-4 mt-12">
                            <div className="h-1.5 w-40 bg-gradient-to-r from-accent-primary via-accent-secondary to-transparent rounded-full shadow-glow" />
                            <div className="h-1 w-20 bg-gradient-to-r from-accent-secondary/50 to-transparent rounded-full" />
                        </div>
                    </div>
                </header>

                {/* Chapter Grid */}
                <div className="max-w-7xl mx-auto px-6">
                    {/* Section Title */}
                    <div className="mb-12">
                        <h2 className="text-3xl font-bold text-text-primary mb-3 flex items-center gap-3">
                            <span className="w-2 h-8 bg-gradient-to-b from-accent-primary to-accent-secondary rounded-full" />
                            课程章节
                        </h2>
                        <p className="text-text-tertiary text-lg ml-5">选择章节开始学习旅程</p>
                    </div>

                    <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
                        {chapters.map((chapter, index) => (
                            <Link
                                key={chapter.id}
                                href={`/${params.module}/${chapter.id}`}
                                className="group relative bg-gradient-to-br from-bg-elevated via-bg-elevated to-bg-base border-2 border-border-subtle/60 rounded-3xl p-8 hover:border-accent-primary/70 hover:shadow-2xl hover:shadow-accent-primary/10 transition-all duration-500 overflow-hidden backdrop-blur-sm"
                                style={{ animationDelay: `${index * 50}ms` }}
                            >
                                {/* Animated Background Gradient */}
                                <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/0 via-transparent to-accent-secondary/0 group-hover:from-accent-primary/10 group-hover:via-accent-primary/5 group-hover:to-accent-secondary/10 transition-all duration-700 ease-out" />
                                
                                {/* Shine Effect */}
                                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                                    <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white/10 group-hover:animate-shine" />
                                </div>

                                {/* Content Container */}
                                <div className="relative z-10">
                                    {/* Chapter Number Badge */}
                                    <div className="flex items-start justify-between mb-6">
                                        <div className="relative">
                                            <div className="absolute -inset-1 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-2xl opacity-75 group-hover:opacity-100 blur group-hover:blur-md transition-all duration-300" />
                                            <div className="relative w-20 h-20 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-2xl flex items-center justify-center text-white font-black text-3xl shadow-xl group-hover:scale-110 group-hover:-rotate-6 transition-all duration-500">
                                                {index + 1}
                                            </div>
                                        </div>
                                        <div className="flex flex-col items-end gap-1.5">
                                            <span className="text-xs font-bold text-accent-primary/80 uppercase tracking-widest bg-accent-primary/10 px-3 py-1 rounded-full">
                                                Chapter
                                            </span>
                                        </div>
                                    </div>

                                    {/* Chapter Title */}
                                    <h3 className="text-2xl font-black text-text-primary mb-4 group-hover:text-accent-primary transition-colors duration-300 leading-tight line-clamp-2 min-h-[3.5rem]">
                                        {chapter.title}
                                    </h3>
                                    
                                    {/* Description */}
                                    {chapter.description && (
                                        <p className="text-base text-text-tertiary leading-relaxed line-clamp-3 mb-6 min-h-[4.5rem]">
                                            {chapter.description}
                                        </p>
                                    )}
                                    
                                    {/* Divider */}
                                    <div className="h-px bg-gradient-to-r from-border-subtle via-accent-primary/20 to-border-subtle mb-6 group-hover:from-accent-primary/50 group-hover:to-accent-secondary/50 transition-all duration-500" />

                                    {/* CTA Button */}
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3 text-accent-primary font-bold text-base group-hover:gap-4 transition-all duration-300">
                                            <span className="relative">
                                                开始学习
                                                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-accent-primary group-hover:w-full transition-all duration-300" />
                                            </span>
                                            <svg className="w-6 h-6 group-hover:translate-x-2 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                            </svg>
                                        </div>
                                    </div>
                                </div>

                                {/* Corner Accent */}
                                <div className="absolute -top-8 -right-8 w-32 h-32 bg-gradient-to-br from-accent-primary/20 via-accent-secondary/10 to-transparent rounded-full opacity-0 group-hover:opacity-100 blur-2xl transition-all duration-500" />
                                <div className="absolute -bottom-8 -left-8 w-32 h-32 bg-gradient-to-tr from-accent-secondary/20 via-accent-primary/10 to-transparent rounded-full opacity-0 group-hover:opacity-100 blur-2xl transition-all duration-500" />
                            </Link>
                        ))}
                    </div>

                    {/* Bottom CTA Section */}
                    <div className="mt-20 text-center">
                        <div className="inline-flex flex-col items-center gap-4 p-8 bg-gradient-to-br from-accent-primary/5 to-accent-secondary/5 rounded-3xl border border-accent-primary/20">
                            <svg className="w-12 h-12 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            <p className="text-lg text-text-secondary font-medium">
                                准备好开启学习之旅了吗？选择一个章节开始吧！
                            </p>
                        </div>
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
