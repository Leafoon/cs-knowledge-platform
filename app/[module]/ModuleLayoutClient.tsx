"use client";

import { ReactNode, useState } from "react";
import { ScrollProgress } from "@/components/knowledge/ScrollProgress";
import { TOC } from "@/components/knowledge/TOC";
import { ModuleRightSidebar } from "@/components/knowledge/ModuleRightSidebar";
import { useScrollSpy } from "@/hooks/useScrollSpy";
import { motion, AnimatePresence } from "framer-motion";

interface ModuleLayoutClientProps {
    children: ReactNode;
    toc: any[];
}

export function ModuleLayoutClient({ children, toc }: ModuleLayoutClientProps) {
    const activeId = useScrollSpy(["h1", "h2", "h3"]);
    const [isMobileTOCOpen, setIsMobileTOCOpen] = useState(false);

    return (
        <>
            <ScrollProgress />

            {/* Mobile TOC Toggle Button */}
            <button
                onClick={() => setIsMobileTOCOpen(true)}
                className="xl:hidden fixed bottom-6 right-6 z-40 w-14 h-14 bg-accent-primary text-white rounded-full shadow-lg flex items-center justify-center hover:scale-110 transition-transform active:scale-95"
                aria-label="Open table of contents"
            >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
            </button>

            {/* Mobile TOC Drawer */}
            <AnimatePresence>
                {isMobileTOCOpen && (
                    <>
                        {/* Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setIsMobileTOCOpen(false)}
                            className="xl:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
                        />

                        {/* Drawer */}
                        <motion.div
                            initial={{ x: "100%" }}
                            animate={{ x: 0 }}
                            exit={{ x: "100%" }}
                            transition={{ type: "spring", damping: 25, stiffness: 200 }}
                            className="xl:hidden fixed right-0 top-0 bottom-0 w-80 max-w-[85vw] bg-bg-elevated border-l border-border-subtle shadow-2xl z-50 overflow-y-auto"
                        >
                            {/* Close Button */}
                            <div className="sticky top-0 bg-bg-elevated border-b border-border-subtle p-4 flex items-center justify-between">
                                <h3 className="text-lg font-semibold text-text-primary">目录</h3>
                                <button
                                    onClick={() => setIsMobileTOCOpen(false)}
                                    className="p-2 hover:bg-bg-base rounded-lg transition-colors"
                                    aria-label="Close"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                            <div className="p-4">
                                <TOC items={toc} activeId={activeId} />
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>

            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-12">
                <div className="flex gap-8 lg:gap-12 xl:gap-16 justify-center lg:justify-start max-w-[1600px] mx-auto">
                    {/* Desktop Sidebar TOC */}
                    <aside className="hidden xl:block w-72 shrink-0 sticky top-20 self-start h-[calc(100vh-6rem)] overflow-y-auto scrollbar-thin scrollbar-thumb-accent-primary/20 scrollbar-track-transparent">
                        <TOC items={toc} activeId={activeId} />
                    </aside>

                    {/* Main Content */}
                    <div className="flex-1 max-w-4xl w-full min-w-0">
                        {children}
                    </div>

                    {/* Right sidebar space for future expansion */}
                    <aside className="hidden 2xl:block w-64 shrink-0 relative">
                        {/* Reserved for related content, metadata, etc. */}
                        <ModuleRightSidebar currentSection={activeId} />
                    </aside>
                </div>
            </div>
        </>
    );
}
