"use client";

import { useEffect, useState } from "react";

export function useScrollSpy(selectors: string[]) {
    const [activeId, setActiveId] = useState<string>("");

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveId(entry.target.id);
                    }
                });
            },
            {
                rootMargin: "-80px 0px -80% 0px",
                threshold: 0.1,
            }
        );

        // Observe all heading elements
        selectors.forEach((selector) => {
            const elements = document.querySelectorAll(selector);
            elements.forEach((el) => observer.observe(el));
        });

        return () => observer.disconnect();
    }, [selectors]);

    return activeId;
}
