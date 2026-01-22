"use client";

import { useState } from "react";
import { TOCItem } from "@/types/content";
import { cn } from "@/lib/utils";

interface TOCProps {
    items: TOCItem[];
    activeId: string;
}

export function TOC({ items, activeId }: TOCProps) {
    const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set(items.map(item => item.id)));

    const scrollToSection = (id: string) => {
        const element = document.getElementById(id);
        if (element) {
            const offset = 80;
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: "smooth",
            });
        }
    };

    const toggleExpand = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setExpandedItems(prev => {
            const newSet = new Set(prev);
            if (newSet.has(id)) {
                newSet.delete(id);
            } else {
                newSet.add(id);
            }
            return newSet;
        });
    };

    const hasChildren = (item: TOCItem) => {
        return item.children && item.children.length > 0;
    };

    const renderTOCItem = (item: TOCItem, depth: number = 0) => {
        const isExpanded = expandedItems.has(item.id);
        const isActive = activeId === item.id;
        const children = item.children || [];

        return (
            <li key={item.id}>
                <div className="relative group">
                    <div
                        className={cn(
                            "w-full flex items-center justify-between text-sm py-2.5 px-3 rounded-xl transition-all duration-200",
                            "border-l-3 relative overflow-hidden",
                            isActive
                                ? "border-accent-primary text-accent-primary font-bold bg-gradient-to-r from-accent-primary/15 to-accent-primary/5 shadow-sm"
                                : "border-transparent text-text-secondary hover:text-text-primary hover:bg-gradient-to-r hover:from-bg-elevated hover:to-transparent"
                        )}
                        style={{ paddingLeft: `${depth * 12 + 12}px` }}
                    >
                        {/* Active indicator */}
                        {isActive && (
                            <div className="absolute left-1 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-accent-primary" />
                        )}

                        {/* Text - Click to Scroll */}
                        <button
                            onClick={() => scrollToSection(item.id)}
                            className="relative z-10 flex-1 text-left line-clamp-2 focus:outline-none"
                        >
                            {item.title}
                        </button>

                        {/* Expand/Collapse Icon - Click to Toggle */}
                        {hasChildren(item) && (
                            <button
                                onClick={(e) => toggleExpand(item.id, e)}
                                className="relative z-10 ml-2 p-1 hover:bg-accent-primary/10 rounded-full transition-colors flex-shrink-0"
                            >
                                <svg
                                    className={cn(
                                        "w-4 h-4 transition-transform duration-200",
                                        isExpanded ? "rotate-90" : ""
                                    )}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        )}
                    </div>
                </div>

                {/* Children with smooth Grid animation */}
                {hasChildren(item) && (
                    <div
                        className={cn(
                            "grid transition-[grid-template-rows] duration-200 ease-out",
                            isExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
                        )}
                    >
                        <div className="overflow-hidden">
                            <ul className={cn(
                                "transition-opacity duration-200",
                                isExpanded ? "opacity-100" : "opacity-0"
                            )}>
                                {children.map(child => renderTOCItem(child, depth + 1))}
                            </ul>
                        </div>
                    </div>
                )}
            </li>
        );
    };

    return (
        <nav className="sticky top-20 max-h-[calc(100vh-5rem)] overflow-hidden">
            {/* TOC Container - Performance optimized */}
            <div className="relative bg-bg-elevated/95 border border-border-subtle/50 rounded-2xl shadow-lg overflow-hidden" style={{ willChange: 'transform' }}>
                {/* Simplified overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/3 to-transparent pointer-events-none" />
                
                <div className="relative p-6">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6 pb-4 border-b border-border-subtle/50">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary/20 to-accent-secondary/20 flex items-center justify-center">
                                <svg className="w-4 h-4 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                                </svg>
                            </div>
                            <h3 className="text-sm font-bold text-text-primary uppercase tracking-wide">
                                本章目录
                            </h3>
                        </div>

                        {/* Expand/Collapse All */}
                        <button
                            onClick={() => {
                                if (expandedItems.size === 0) {
                                    const allIds = new Set<string>();
                                    const traverse = (list: TOCItem[]) => {
                                        list.forEach(item => {
                                            allIds.add(item.id);
                                            if (item.children) traverse(item.children);
                                        });
                                    };
                                    traverse(items);
                                    setExpandedItems(allIds);
                                } else {
                                    setExpandedItems(new Set());
                                }
                            }}
                            className="text-xs text-accent-primary hover:text-accent-secondary transition-colors font-semibold flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-accent-primary/10"
                        >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                {expandedItems.size === 0 ? (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                ) : (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                                )}
                            </svg>
                            {expandedItems.size === 0 ? "全部展开" : "全部收起"}
                        </button>
                    </div>

                    {/* Scrollable Content */}
                    <div className="overflow-y-auto scrollbar-thin scrollbar-thumb-accent-primary/20 scrollbar-track-transparent pr-2 max-h-[calc(100vh-13rem)]">
                        <ul className="space-y-1">
                            {items.map(item => renderTOCItem(item, 0))}
                        </ul>
                    </div>

                    {/* Footer hint */}
                    <div className="mt-4 pt-4 border-t border-border-subtle/30">
                        <div className="flex items-center gap-2 text-xs text-text-tertiary/80">
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                            </svg>
                            <span>点击章节快速跳转</span>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
}
