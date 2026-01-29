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
        const isTopLevel = depth === 0;

        return (
            <li key={item.id} className={cn(isTopLevel && "mb-1")}>
                <div className="relative group">
                    <div
                        className={cn(
                            "w-full flex items-center justify-between rounded-lg transition-all duration-200 relative cursor-pointer",
                            isTopLevel ? "py-2.5 px-3.5" : "py-2 px-3",
                            isActive
                                ? "text-accent-primary font-semibold bg-gradient-to-r from-accent-primary/12 to-accent-primary/5 shadow-sm border-l-2 border-accent-primary"
                                : "text-text-secondary hover:text-text-primary hover:bg-bg-base/60 border-l-2 border-transparent",
                            isTopLevel && !isActive && "font-medium text-text-primary"
                        )}
                        style={{ paddingLeft: `${depth * 14 + (isActive ? 10 : 12)}px` }}
                    >

                        {/* Collapse icon for items with children */}
                        {hasChildren(item) && (
                            <button
                                onClick={(e) => toggleExpand(item.id, e)}
                                className="mr-2 p-1 hover:bg-accent-primary/15 rounded-md transition-all flex-shrink-0"
                                aria-label={isExpanded ? "Collapse" : "Expand"}
                            >
                                <svg
                                    className={cn(
                                        "w-3.5 h-3.5 transition-transform duration-200",
                                        isExpanded ? "rotate-90" : "",
                                        isActive ? "text-accent-primary" : "text-text-tertiary"
                                    )}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        )}

                        {/* Text - Click to Scroll */}
                        <button
                            onClick={() => scrollToSection(item.id)}
                            className={cn(
                                "flex-1 text-left focus:outline-none transition-all",
                                isTopLevel ? "text-[13px] leading-snug" : "text-[13px] leading-snug",
                                isActive && "tracking-wide"
                            )}
                        >
                            <span className="line-clamp-2 break-words">{item.title}</span>
                        </button>

                        {/* Chapter number badge for top-level items */}
                        {isTopLevel && item.title.match(/Chapter\s+(\d+)/i) && (
                            <span className={cn(
                                "ml-auto pl-2 text-[11px] font-semibold rounded flex-shrink-0 transition-all",
                                isActive
                                    ? "text-accent-primary"
                                    : "text-text-tertiary/60 group-hover:text-accent-primary/80"
                            )}>
                                #{item.title.match(/Chapter\s+(\d+)/i)?.[1]}
                            </span>
                        )}
                    </div>
                </div>

                {/* Children with smooth collapse animation */}
                {hasChildren(item) && (
                    <div
                        className={cn(
                            "grid transition-[grid-template-rows] duration-200 ease-out",
                            isExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
                        )}
                    >
                        <div className="overflow-hidden">
                            <ul className={cn(
                                "mt-0.5 space-y-0.5 transition-opacity duration-150",
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
            {/* Professional TOC Container */}
            <div className="relative bg-bg-elevated border border-border-subtle rounded-2xl shadow-soft overflow-hidden">
                <div className="relative">
                    {/* Professional Header */}
                    <div className="flex items-center justify-between px-5 py-4 border-b border-border-subtle bg-gradient-to-b from-bg-base/50 to-transparent">
                        <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2">
                                <svg className="w-5 h-5 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h8M4 18h16" />
                                </svg>
                                <h3 className="text-sm font-semibold text-text-primary">
                                    本章目录
                                </h3>
                            </div>
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
                            className="text-xs text-text-tertiary hover:text-accent-primary transition-colors font-medium flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg hover:bg-accent-primary/8 border border-transparent hover:border-accent-primary/20"
                            aria-label={expandedItems.size === 0 ? "Expand all" : "Collapse all"}
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                {expandedItems.size === 0 ? (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                ) : (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                                )}
                            </svg>
                            <span>{expandedItems.size === 0 ? "展开" : "收起"}</span>
                        </button>
                    </div>

                    {/* Scrollable Content */}
                    <div className="overflow-y-auto scrollbar-thin scrollbar-thumb-accent-primary/25 hover:scrollbar-thumb-accent-primary/50 scrollbar-track-transparent px-3 py-4 max-h-[calc(100vh-12rem)]" style={{ scrollbarGutter: 'stable' }}>
                        <ul className="space-y-1">
                            {items.map(item => renderTOCItem(item, 0))}
                        </ul>
                    </div>

                    {/* Footer Hint */}
                    <div className="px-5 py-3 border-t border-border-subtle bg-gradient-to-t from-bg-base/30 to-transparent">
                        <div className="flex items-center gap-2 text-xs text-text-tertiary/90">
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 9l3 3m0 0l-3 3m3-3H8m13 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <span className="font-medium">点击章节快速跳转</span>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
}
