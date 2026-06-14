"use client";

import { useEffect, useState } from "react";
import { Command } from "cmdk";
import { useSearch } from "./SearchContext";
import { useRouter } from "next/navigation";
import { Search, FileText, Layers, Hash } from "lucide-react";

import { Module } from "@/types/content";

// Define a simplified Lucide icon type or just use any
type IconType = any;

interface CommandMenuProps {
    modules: Module[];
}

export function CommandMenu({ modules }: CommandMenuProps) {
    const { isOpen, setIsOpen } = useSearch();
    const router = useRouter();
    const [query, setQuery] = useState("");

    // Reset query when opening
    useEffect(() => {
        if (!isOpen) {
            setQuery("");
        }
    }, [isOpen]);

    const runCommand = (command: () => void) => {
        setIsOpen(false);
        command();
    };

    return (
        <Command.Dialog
            open={isOpen}
            onOpenChange={setIsOpen}
            label="Global Command Menu"
            className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-black/50 backdrop-blur-sm dark:bg-black/80"
        >
            <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl overflow-hidden w-full max-w-2xl border border-slate-200 dark:border-slate-800 animate-in fade-in zoom-in-95 duration-100">
                <div className="flex items-center border-b border-slate-100 dark:border-slate-800 px-4">
                    <Search className="w-5 h-5 text-slate-400 mr-2" />
                    <Command.Input
                        value={query}
                        onValueChange={setQuery}
                        placeholder="搜索知识模块..."
                        className="w-full h-14 bg-transparent outline-none text-slate-700 dark:text-slate-200 placeholder:text-slate-400 text-base"
                    />
                </div>

                <Command.List className="max-h-[300px] overflow-y-auto overflow-x-hidden py-2 scrollbar-thin">
                    <Command.Empty className="py-6 text-center text-sm text-slate-500 dark:text-slate-400">
                        未找到相关结果
                    </Command.Empty>

                    <Command.Group heading="知识模块" className="text-xs font-semibold text-slate-500 dark:text-slate-400 px-2 py-1.5">
                        {modules.map((module) => (
                            <Command.Item
                                key={module.id}
                                value={module.title + " " + module.description}
                                onSelect={() => {
                                    runCommand(() => router.push(`/${module.id}`));
                                }}
                                className="flex items-center px-4 py-3 text-sm text-slate-700 dark:text-slate-200 cursor-pointer rounded-lg mx-2 aria-selected:bg-indigo-50 dark:aria-selected:bg-indigo-900/20 aria-selected:text-indigo-600 dark:aria-selected:text-indigo-300 transition-colors"
                            >
                                <div className="w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center bg-slate-100 dark:bg-slate-800 mr-3 text-lg">
                                    {/* Ideally map icon here, for now use generic */}
                                    <Layers className="w-4 h-4" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium truncate">{module.title}</div>
                                    <div className="text-xs text-slate-400 truncate">{module.description || "无描述"}</div>
                                </div>
                            </Command.Item>
                        ))}
                    </Command.Group>

                    {/* Placeholder for 'Chapters' if we had them indexed flatly */}
                    {/* <Command.Group heading="核心概念"> ... </Command.Group> */}
                </Command.List>

                <div className="border-t border-slate-100 dark:border-slate-800 px-4 py-2.5 bg-slate-50 dark:bg-slate-900/50 flex items-center justify-between">
                    <span className="text-xs text-slate-400">
                        Tip: 使用 <kbd className="font-sans px-1 bg-white dark:bg-slate-800 border dark:border-slate-700 rounded">↑</kbd> <kbd className="font-sans px-1 bg-white dark:bg-slate-800 border dark:border-slate-700 rounded">↓</kbd> 导航，<kbd className="font-sans px-1 bg-white dark:bg-slate-800 border dark:border-slate-700 rounded">Enter</kbd> 选择
                    </span>
                    <span className="text-xs text-slate-400">
                        CS Knowledge Platform
                    </span>
                </div>
            </div>
        </Command.Dialog>
    );
}
