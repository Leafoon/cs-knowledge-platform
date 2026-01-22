import { HTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
    variant?: "default" | "success" | "warning" | "info";
}

const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
    ({ className, variant = "default", ...props }, ref) => {
        return (
            <span
                ref={ref}
                className={cn(
                    "inline-flex items-center rounded-md px-2 py-1 text-xs font-medium border",
                    "transition-colors duration-200",
                    {
                        "bg-accent-primary/5 text-accent-primary border-accent-primary/20": variant === "default",
                        "bg-green-500/5 text-green-700 dark:text-green-400 border-green-500/20": variant === "success",
                        "bg-yellow-500/5 text-yellow-700 dark:text-yellow-400 border-yellow-500/20": variant === "warning",
                        "bg-blue-500/5 text-blue-700 dark:text-blue-400 border-blue-500/20": variant === "info",
                    },
                    className
                )}
                {...props}
            />
        );
    }
);

Badge.displayName = "Badge";

export { Badge };
