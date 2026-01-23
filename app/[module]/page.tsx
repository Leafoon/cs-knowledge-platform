import { getModuleContent, getModules } from "@/lib/content-loader";
import { ContentRenderer } from "@/components/knowledge/ContentRenderer";
import { Badge } from "@/components/ui/Badge";
import { Suspense } from "react";

export async function generateStaticParams() {
    const modules = getModules();
    return modules.map((module) => ({
        module: module.id,
    }));
}

// Loading component
function ContentLoading() {
    return (
        <div className="flex items-center justify-center py-20">
            <div className="flex flex-col items-center gap-4">
                <div className="w-12 h-12 border-4 border-accent-primary/30 border-t-accent-primary rounded-full animate-spin" />
                <p className="text-text-secondary text-lg font-medium">加载内容中...</p>
            </div>
        </div>
    );
}

export default async function ModulePage({
    params,
}: {
    params: { module: string };
}) {
    const { html, frontmatter } = await getModuleContent(params.module);
    const modules = getModules();
    const currentModule = modules.find((m) => m.id === params.module);

    return (
        <article className="min-h-screen">
            {/* Module Header */}
            <header className="relative mb-16 pb-12 overflow-hidden">
                {/* Simplified Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/5 to-transparent -z-10" />

                {/* Content */}
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
                    
                    {/* Decorative separator */}
                    <div className="mt-10 h-1 w-24 bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full" />
                </div>
            </header>

            {/* Content with embedded interactive components */}
            <ContentRenderer html={html} moduleId={params.module} />
        </article>
    );
}
