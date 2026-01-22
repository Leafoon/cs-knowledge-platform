"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/Button";

export function Header() {
    const [isDark, setIsDark] = useState(false);

    useEffect(() => {
        // Check initial theme
        const isDarkMode = document.documentElement.classList.contains('dark');
        setIsDark(isDarkMode);
    }, []);

    const toggleTheme = () => {
        const newIsDark = !isDark;
        setIsDark(newIsDark);
        document.documentElement.classList.toggle('dark', newIsDark);
        localStorage.setItem('theme', newIsDark ? 'dark' : 'light');
    };

    return (
        <header className="sticky top-0 z-50 w-full border-b border-border-subtle bg-bg-elevated/80 backdrop-blur-sm">
            <div className="container mx-auto flex h-16 items-center justify-between px-6">
                <Link href="/" className="flex items-center space-x-3 group">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center transition-transform group-hover:scale-110">
                        <span className="text-white font-bold text-lg">CS</span>
                    </div>
                    <span className="text-lg font-semibold text-text-primary">
                        Knowledge Platform
                    </span>
                </Link>

                <nav className="flex items-center space-x-4">
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={toggleTheme}
                        aria-label="Toggle theme"
                    >
                        {isDark ? (
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                            </svg>
                        ) : (
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                            </svg>
                        )}
                    </Button>
                </nav>
            </div>
        </header>
    );
}
