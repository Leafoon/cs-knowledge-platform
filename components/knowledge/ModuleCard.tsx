"use client";

import Link from "next/link";
import { Module } from "@/types/content";
import {
    Cpu,
    Terminal,
    Flame,
    Network,
    Database,
    Coffee,
    Binary,
    BookOpen,
    Bot,
    Box,
    Layers,
    MoveRight,
    LucideIcon
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ModuleCardProps {
    module: Module;
}

export function ModuleCard({ module }: ModuleCardProps) {
    const CardWrapper = module.externalLink ? "a" : Link;
    const linkProps = module.externalLink
        ? { href: module.externalLink, target: "_blank", rel: "noopener noreferrer" }
        : { href: `/${module.id}` };

    const Icon = getModuleIcon(module.id);

    return (
        <CardWrapper {...linkProps} className="block h-full group">
            <div className={cn(
                "relative h-full flex flex-col p-6 rounded-2xl transition-all duration-300",
                "bg-white dark:bg-slate-900/50 backdrop-blur-sm",
                "border border-slate-200 dark:border-slate-800",
                "hover:border-slate-300 dark:hover:border-slate-700",
                "hover:shadow-xl hover:shadow-slate-200/50 dark:hover:shadow-black/50",
                "hover:-translate-y-1"
            )}>
                {/* Background Glow Effect on Hover */}
                <div
                    className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-5 transition-opacity duration-500 pointer-events-none"
                    style={{ backgroundColor: module.color }}
                />

                <div className="flex items-start justify-between mb-6 relative z-10">
                    {/* Icon Box */}
                    <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110 group-hover:rotate-3 shadow-sm"
                        style={{
                            backgroundColor: `${module.color}15`,
                            color: module.color,
                            boxShadow: `inset 0 0 0 1px ${module.color}30`
                        }}
                    >
                        <Icon className="w-6 h-6" strokeWidth={2.5} />
                    </div>

                    {/* Arrow Indicator */}
                    <div className="text-slate-300 group-hover:text-slate-500 dark:text-slate-700 dark:group-hover:text-slate-400 transition-colors">
                        <MoveRight className="w-5 h-5 -rotate-45 group-hover:rotate-0 transition-transform duration-300" />
                    </div>
                </div>

                <div className="relative z-10 flex-1">
                    <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-2 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                        {module.title}
                    </h3>

                    <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed line-clamp-2">
                        {module.description}
                    </p>
                </div>

                {/* Footer Info */}
                <div className="mt-6 pt-4 border-t border-slate-100 dark:border-slate-800/50 flex items-center justify-between relative z-10">
                    {module.chapters && (
                        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-300 transition-colors">
                            <Layers className="w-3.5 h-3.5" />
                            <span>{module.chapters.length} 章节</span>
                        </div>
                    )}

                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors opacity-0 group-hover:opacity-100 translate-x-2 group-hover:translate-x-0 transition-all duration-300">
                        <span>开始学习</span>
                    </div>
                </div>
            </div>
        </CardWrapper>
    );
}

function getModuleIcon(id: string): LucideIcon {
    const icons: Record<string, LucideIcon> = {
        "operating-system": Cpu,
        "python": Terminal,
        "pytorch": Flame,
        "computer-network": Network,
        "database": Database,
        "java": Coffee,
        "linux-commands": Terminal,
        "computer-organization": Binary,
        "hugging-face": Bot,
        "langchain": Box,
        "reinforcement-learning": Bot,
    };
    return icons[id] || BookOpen;
}
