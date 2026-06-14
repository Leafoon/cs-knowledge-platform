"use client";

import { useEffect, useRef, useState } from "react";

export function useScrollSpy(ids: string[]) {
    const [activeId, setActiveId] = useState<string>("");
    // Keep a stable ref to activeId for use inside the observer callback
    const activeIdRef = useRef(activeId);
    activeIdRef.current = activeId;

    useEffect(() => {
        if (ids.length === 0) return;

        // Track all currently-intersecting headings to pick the topmost one
        const intersectingSet = new Set<string>();

        const pickActive = () => {
            // Find the topmost intersecting heading (by DOM order matches ids order)
            for (const id of ids) {
                if (intersectingSet.has(id)) {
                    setActiveId(id);
                    return;
                }
            }
            // If nothing intersects (gap between sections), keep last active —
            // only clear if we have a truly fresh page (no prior active).
        };

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        intersectingSet.add(entry.target.id);
                    } else {
                        intersectingSet.delete(entry.target.id);
                    }
                });
                pickActive();
            },
            {
                // Top 30% of viewport — wider window reduces "nothing active" gaps
                rootMargin: "-80px 0px -70% 0px",
                threshold: 0,
            }
        );

        ids.forEach((id) => {
            const element = document.getElementById(id);
            if (element) observer.observe(element);
        });

        return () => observer.disconnect();
    }, [ids]);

    return activeId;
}
