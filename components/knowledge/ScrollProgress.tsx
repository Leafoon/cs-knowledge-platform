"use client";

import { useEffect, useState } from "react";

export function ScrollProgress() {
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        const updateProgress = () => {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            setProgress(scrollPercent);
        };

        window.addEventListener("scroll", updateProgress);
        updateProgress();

        return () => window.removeEventListener("scroll", updateProgress);
    }, []);

    return (
        <div className="fixed top-16 left-0 right-0 z-40 h-1 bg-border-subtle/30">
            <div
                className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary origin-left"
                style={{ 
                    transform: `scaleX(${progress / 100})`,
                    transition: 'transform 150ms ease-out',
                    willChange: 'transform'
                }}
            />
        </div>
    );
}
