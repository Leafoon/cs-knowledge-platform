"use client";

import { Module } from "@/types/content";
import { ModuleCard } from "./ModuleCard";

interface ModuleGridProps {
    modules: Module[];
}

export function ModuleGrid({ modules }: ModuleGridProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {modules.map((module, index) => (
                <div
                    key={module.id}
                    className="animate-fade-in"
                    style={{
                        animationDelay: `${index * 30}ms`,
                    }}
                >
                    <ModuleCard module={module} />
                </div>
            ))}
        </div>
    );
}
