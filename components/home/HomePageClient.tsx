"use client";

import { ModuleGrid } from "@/components/knowledge/ModuleGrid";
import { Search, Sparkles, TrendingUp, Zap } from "lucide-react";
import { useSearch } from "@/components/search/SearchContext";
import { Module } from "@/types/content";
import { motion } from "framer-motion";

interface HomePageClientProps {
    modules: Module[];
}

export function HomePageClient({ modules }: HomePageClientProps) {
    const { setIsOpen } = useSearch();

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50/20 dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950/10 selection:bg-indigo-100 dark:selection:bg-indigo-900/40 font-sans">
            {/* Optimized Background - Reduced GPU Load */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                {/* Simplified Grid Pattern */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:32px_32px] opacity-40"></div>

                {/* Reduced Gradient Orbs - Less Blur */}
                <div className="absolute left-1/2 top-[-10%] -translate-x-1/2 -z-10 h-[600px] w-[600px] bg-gradient-to-br from-indigo-400/20 via-purple-400/15 to-pink-400/10 dark:from-indigo-500/12 dark:via-purple-500/8 dark:to-pink-500/5 blur-[80px] rounded-full"></div>
                <div className="absolute right-[20%] top-[20%] -z-10 h-[400px] w-[400px] bg-gradient-to-br from-cyan-400/15 to-blue-400/10 dark:from-cyan-500/10 dark:to-blue-500/6 blur-[60px] rounded-full"></div>
            </div>

            <div className="relative container mx-auto px-4 sm:px-6 lg:px-8 pt-20 sm:pt-28 pb-16">
                {/* Hero Section */}
                <section className="text-center max-w-6xl mx-auto mb-28">
                    {/* Simplified Badge */}
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                        className="inline-flex items-center gap-2.5 px-5 py-2 rounded-full bg-white/90 dark:bg-slate-900/90 border border-indigo-200/60 dark:border-indigo-800/40 mb-12 shadow-md backdrop-blur-sm group cursor-default"
                    >
                        <span className="relative flex h-2.5 w-2.5">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500"></span>
                        </span>
                        <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                        <span className="text-sm font-bold text-indigo-900 dark:text-indigo-100 tracking-wide">
                            CS Knowledge Platform
                        </span>
                    </motion.div>

                    {/* Simplified Title */}
                    <motion.h1
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                        className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black text-slate-900 dark:text-slate-50 mb-8 leading-[1.1] tracking-tight"
                    >
                        <span className="block mb-2">构建你的</span>
                        <span className="relative inline-block">
                            {/* Simplified Glow - Single Layer */}
                            <span className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 opacity-20 dark:opacity-15 blur-2xl"></span>

                            {/* Main Gradient Text */}
                            <span className="relative text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 dark:from-indigo-400 dark:via-purple-400 dark:to-pink-400">
                                计算机科学
                            </span>
                        </span>
                        <span className="block mt-2">核心知识体系</span>
                    </motion.h1>

                    {/* Subtitle */}
                    <motion.p
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        className="text-xl sm:text-2xl md:text-3xl text-slate-600 dark:text-slate-300 mb-14 max-w-4xl mx-auto leading-relaxed font-medium tracking-tight"
                    >
                        <span className="block mb-2">涵盖操作系统、人工智能、后端架构等核心领域。</span>
                        <span className="block text-lg sm:text-xl text-slate-500 dark:text-slate-400">
                            源于一线大厂实践，打造专业的计算机技术能力。
                        </span>
                    </motion.p>

                    {/* Optimized Search Bar */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.3 }}
                        className="relative max-w-2xl mx-auto mb-20"
                    >
                        <div
                            className="group relative cursor-pointer"
                            onClick={() => setIsOpen(true)}
                        >
                            {/* Simplified Hover Effect */}
                            <div className="relative flex items-center bg-white/90 dark:bg-slate-900/90 backdrop-blur-sm rounded-2xl px-6 py-5 shadow-xl ring-1 ring-slate-200/50 dark:ring-slate-700/50 hover:ring-indigo-500/40 dark:hover:ring-indigo-400/40 transition-all duration-300 hover:shadow-2xl">
                                <Search className="h-6 w-6 text-slate-400 dark:text-slate-500 mr-4 group-hover:text-indigo-500 dark:group-hover:text-indigo-400 transition-colors" />
                                <span className="flex-1 text-slate-400 dark:text-slate-500 text-lg font-medium">
                                    搜索知识模块、核心概念...
                                </span>
                                <div className="flex items-center gap-2">
                                    <kbd className="hidden sm:inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 px-3 text-sm font-semibold text-slate-500 dark:text-slate-400 shadow-sm">
                                        <span className="text-base">⌘</span> K
                                    </kbd>
                                </div>
                            </div>
                        </div>
                    </motion.div>

                    {/* Simplified Stats Section */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.4 }}
                        className="flex flex-wrap items-center justify-center gap-x-20 gap-y-10"
                    >
                        <StatsItem value={modules.length} label="知识模块" icon={Zap} />
                        <div className="w-px h-16 bg-gradient-to-b from-transparent via-slate-300 dark:via-slate-700 to-transparent hidden sm:block"></div>
                        <StatsItem value="∞" label="持续更新" icon={TrendingUp} />
                        <div className="w-px h-16 bg-gradient-to-b from-transparent via-slate-300 dark:via-slate-700 to-transparent hidden sm:block"></div>
                        <StatsItem value="100%" label="完全开源" icon={Sparkles} />
                    </motion.div>
                </section>

                {/* Module Grid Section */}
                <motion.section
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.5 }}
                    className="max-w-7xl mx-auto"
                >
                    <div className="flex items-end justify-between mb-14 px-2">
                        <div>
                            <h2 className="text-4xl font-black text-slate-900 dark:text-slate-50 mb-3 tracking-tight">
                                探索知识模块
                            </h2>
                            <p className="text-slate-500 dark:text-slate-400 text-lg font-medium">
                                选择你感兴趣的领域，开启深度学习之旅
                            </p>
                        </div>
                    </div>
                    <ModuleGrid modules={modules} />
                </motion.section>
            </div>
        </div>
    );
}

function StatsItem({ value, label, icon: Icon }: { value: string | number; label: string; icon: any }) {
    return (
        <div className="text-center group cursor-default">
            {/* Simplified Icon Container */}
            <div className="relative w-14 h-14 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950/40 dark:to-purple-950/40 flex items-center justify-center border border-indigo-100 dark:border-indigo-900/50 group-hover:border-indigo-300 dark:group-hover:border-indigo-700 transition-colors duration-300">
                <Icon className="w-6 h-6 text-indigo-600 dark:text-indigo-400 group-hover:scale-110 transition-transform duration-300" />
            </div>

            {/* Value */}
            <div className="text-5xl font-black bg-gradient-to-br from-slate-900 to-slate-700 dark:from-slate-50 dark:to-slate-300 bg-clip-text text-transparent mb-2 group-hover:from-indigo-600 group-hover:to-purple-600 dark:group-hover:from-indigo-400 dark:group-hover:to-purple-400 transition-all duration-500">
                {value}
            </div>

            {/* Label */}
            <div className="text-sm text-slate-500 dark:text-slate-400 font-bold tracking-wider uppercase group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors duration-300">
                {label}
            </div>
        </div>
    );
}
