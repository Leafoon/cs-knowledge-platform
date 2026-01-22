"use client";

import Link from "next/link";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Module } from "@/types/content";

interface ModuleCardProps {
    module: Module;
}

export function ModuleCard({ module }: ModuleCardProps) {
    const CardWrapper = module.externalLink ? "a" : Link;
    const linkProps = module.externalLink
        ? { href: module.externalLink, target: "_blank", rel: "noopener noreferrer" }
        : { href: `/${module.id}` };

    return (
        <CardWrapper {...linkProps}>
            <Card
                hover
                className="group cursor-pointer relative overflow-hidden backdrop-blur-md bg-bg-elevated/90 border border-border-subtle hover:border-accent-primary/50 transition-all duration-500 hover:shadow-[0_20px_60px_-15px_rgba(99,102,241,0.4)] hover:-translate-y-2 hover:scale-[1.02]"
            >
                {/* Animated Gradient Border */}
                <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700 rounded-lg animate-gradient"
                    style={{
                        background: `linear-gradient(135deg, ${module.color}, transparent, ${module.color})`,
                        backgroundSize: '200% 200%',
                    }}
                />

                {/* Glow Effect */}
                <div
                    className="absolute -inset-1 opacity-0 group-hover:opacity-75 blur-xl transition-opacity duration-500 -z-10"
                    style={{
                        background: `radial-gradient(circle, ${module.color}40, transparent 70%)`,
                    }}
                />

                {/* Top Accent Line - Thicker and More Visible */}
                <div
                    className="absolute top-0 left-0 right-0 h-2 opacity-60 group-hover:opacity-100 transition-all duration-300 group-hover:h-3"
                    style={{
                        background: `linear-gradient(90deg, ${module.color}, ${module.color}cc, ${module.color})`,
                    }}
                />

                <CardHeader className="relative z-10 pt-8">
                    <div className="flex items-start justify-between mb-6">
                        {/* Icon Container - Much Larger and More Prominent */}
                        <div
                            className="relative w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-500 group-hover:scale-125 group-hover:rotate-6 shadow-lg"
                            style={{
                                backgroundColor: `${module.color}20`,
                                boxShadow: `0 0 0 2px ${module.color}30, 0 8px 24px ${module.color}20`,
                            }}
                        >
                            {/* Pulsing Glow */}
                            <div
                                className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl animate-pulse"
                                style={{
                                    backgroundColor: `${module.color}50`,
                                }}
                            />
                            <span className="text-5xl relative z-10 drop-shadow-lg" style={{ color: module.color }}>
                                {getModuleIcon(module.id)}
                            </span>
                        </div>

                        {/* Badge - More Prominent */}
                        <Badge
                            variant="default"
                            className="bg-accent-primary/15 text-accent-primary border-accent-primary/30 font-bold px-4 py-1.5 shadow-sm"
                        >
                            知识模块
                        </Badge>
                    </div>

                    {/* Title - Larger and Bolder */}
                    <CardTitle className="group-hover:text-accent-primary transition-colors duration-300 text-2xl font-extrabold mb-3 leading-tight">
                        {module.title}
                    </CardTitle>

                    {/* Description */}
                    <CardDescription className="text-text-secondary leading-relaxed text-base">
                        {module.description}
                    </CardDescription>
                </CardHeader>

                <CardContent className="relative z-10 pb-6">
                    {module.chapters && module.chapters.length > 0 && (
                        <div className="flex items-center gap-2 text-sm text-text-tertiary bg-bg-base/30 rounded-lg px-4 py-2.5">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                            </svg>
                            <span className="font-semibold">{module.chapters.length} 个章节</span>
                        </div>
                    )}

                    {/* Arrow Icon - Larger and More Animated */}
                    <div className="absolute bottom-6 right-6 opacity-0 group-hover:opacity-100 transform translate-x-4 group-hover:translate-x-0 transition-all duration-500">
                        <div className="bg-accent-primary text-white p-3 rounded-full shadow-lg">
                            <svg
                                className="w-6 h-6"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </CardWrapper >
    );
}

function getModuleIcon(id: string): string {
    const icons: Record<string, string> = {
        "operating-system": "💻",
        "python": "🐍",
        "pytorch": "🔥",
        "computer-network": "🌐",
        "database": "🗄️",
        "java": "☕",
        "linux-commands": "⌨️",
        "computer-organization": "🖥️",
    };
    return icons[id] || "📚";
}
