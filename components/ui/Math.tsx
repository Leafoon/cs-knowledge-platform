"use client";

import katex from "katex";
import { useEffect, useRef } from "react";

interface InlineMathProps {
    children: string;
    className?: string;
}

/**
 * 渲染行内数学公式（inline）
 * 使用 KaTeX 美化渲染
 */
export function InlineMath({ children, className = "" }: InlineMathProps) {
    const ref = useRef<HTMLSpanElement>(null);

    useEffect(() => {
        if (ref.current) {
            try {
                katex.render(children, ref.current, {
                    throwOnError: false,
                    displayMode: false,
                });
            } catch (e) {
                console.error("KaTeX render error:", e);
                ref.current.textContent = children;
            }
        }
    }, [children]);

    return <span ref={ref} className={className} />;
}

/**
 * 渲染块级数学公式（display）
 * 使用 KaTeX 美化渲染
 */
export function DisplayMath({ children, className = "" }: InlineMathProps) {
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (ref.current) {
            try {
                katex.render(children, ref.current, {
                    throwOnError: false,
                    displayMode: true,
                });
            } catch (e) {
                console.error("KaTeX render error:", e);
                ref.current.textContent = children;
            }
        }
    }, [children]);

    return <div ref={ref} className={className} />;
}
