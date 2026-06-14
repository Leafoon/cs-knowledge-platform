"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

interface CopyButtonProps {
    text: string;
    className?: string;
}

export function CopyButton({ text, className }: CopyButtonProps) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error("Failed to copy text: ", err);
        }
    };

    return (
        <button
            onClick={handleCopy}
            className={cn(
                "p-2 rounded-md transition-all duration-200",
                "hover:bg-slate-200 dark:hover:bg-slate-700",
                "focus:outline-none focus:ring-2 focus:ring-indigo-500",
                "text-slate-500 dark:text-slate-400",
                className
            )}
            title="Copy code"
        >
            {copied ? (
                <Check className="w-4 h-4 text-green-500" />
            ) : (
                <Copy className="w-4 h-4" />
            )}
        </button>
    );
}
