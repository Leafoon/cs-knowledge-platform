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

            {/* Mobile TOC Toggle Button - Premium Design */}
            <button
                onClick={() => setIsMobileTOCOpen(true)}
                className="xl:hidden fixed bottom-6 right-6 z-40 w-14 h-14 bg-gradient-to-br from-accent-primary via-accent-primary to-accent-secondary text-white rounded-2xl shadow-soft-lg flex items-center justify-center hover:scale-105 hover:shadow-premium transition-all duration-200 active:scale-95 border border-white/20"
                aria-label="Open table of contents"
            >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h8M4 18h16" />
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
                            className="xl:hidden fixed inset-0 bg-black/70 backdrop-blur-sm z-40"
                        />

                        {/* Drawer */}
                        <motion.div
                            initial={{ x: "100%" }}
                            animate={{ x: 0 }}
                            exit={{ x: "100%" }}
                            transition={{ type: "spring", damping: 35, stiffness: 400 }}
                            className="xl:hidden fixed right-0 top-0 bottom-0 w-80 max-w-[85vw] bg-bg-elevated shadow-premium z-50 overflow-y-auto"
                        >
                            {/* Close Button */}
                            <div className="sticky top-0 bg-gradient-to-b from-bg-elevated to-bg-elevated/95 backdrop-blur-sm border-b border-border-subtle px-5 py-4 flex items-center justify-between z-10">
                                <h3 className="text-base font-semibold text-text-primary flex items-center gap-2">
                                    <svg className="w-5 h-5 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h8M4 18h16" />
                                    </svg>
                                    章节导航
                                </h3>
                                <button
                                    onClick={() => setIsMobileTOCOpen(false)}
                                    className="p-2 hover:bg-bg-base rounded-xl transition-colors text-text-tertiary hover:text-text-primary"
                                    aria-label="Close"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                            <div className="px-4 py-5">
                                <TOC items={toc} activeId={activeId} />
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>

            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-12">
                <div className="flex gap-6 lg:gap-8 xl:gap-10 justify-center lg:justify-start max-w-[1600px] mx-auto">
                    {/* Desktop Sidebar TOC - Modern Design */}
                    <aside className="hidden xl:block w-72 shrink-0 sticky top-20 self-start h-[calc(100vh-6rem)] overflow-y-auto scrollbar-thin scrollbar-thumb-accent-primary/15 scrollbar-track-transparent">
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
