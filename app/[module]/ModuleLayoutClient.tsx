"use client";

import { ReactNode, useMemo } from "react";
import { ScrollProgress } from "@/components/knowledge/ScrollProgress";
import { TOC } from "@/components/knowledge/TOC";
import { useScrollSpy } from "@/hooks/useScrollSpy";

interface ModuleLayoutClientProps {
    children: ReactNode;
    toc: any[];
}

export function ModuleLayoutClient({ children, toc }: ModuleLayoutClientProps) {
    const activeId = useScrollSpy(["h1", "h2", "h3"]);

    return (
        <>
            <ScrollProgress />
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
                <div className="flex gap-8 lg:gap-12 xl:gap-16 justify-center lg:justify-start max-w-[1600px] mx-auto">
                    {/* Sidebar TOC */}
                    <aside className="hidden xl:block w-72 shrink-0 sticky top-20 self-start h-[calc(100vh-6rem)] overflow-y-auto scrollbar-thin scrollbar-thumb-accent-primary/20 scrollbar-track-transparent">
                        <TOC items={toc} activeId={activeId} />
                    </aside>

                    {/* Main Content */}
                    <div className="flex-1 max-w-4xl w-full min-w-0">
                        {children}
                    </div>
                    
                    {/* Right sidebar space for future expansion */}
                    <aside className="hidden 2xl:block w-64 shrink-0">
                        {/* Reserved for related content, metadata, etc. */}
                    </aside>
                </div>
            </div>
        </>
    );
}
