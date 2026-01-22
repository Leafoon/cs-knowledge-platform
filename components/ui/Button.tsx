import { ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "secondary" | "ghost";
    size?: "sm" | "md" | "lg";
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = "primary", size = "md", ...props }, ref) => {
        return (
            <button
                ref={ref}
                className={cn(
                    "inline-flex items-center justify-center rounded-md font-medium transition-all",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary",
                    "disabled:pointer-events-none disabled:opacity-50",
                    {
                        // Variants
                        "bg-gradient-to-r from-accent-primary to-accent-secondary text-white hover:opacity-90 hover:scale-105":
                            variant === "primary",
                        "bg-bg-elevated border border-border-subtle text-text-primary hover:bg-gray-50 dark:hover:bg-gray-800":
                            variant === "secondary",
                        "hover:bg-gray-100 dark:hover:bg-gray-800 text-text-secondary":
                            variant === "ghost",

                        // Sizes
                        "h-8 px-3 text-sm": size === "sm",
                        "h-10 px-4": size === "md",
                        "h-12 px-6 text-lg": size === "lg",
                    },
                    className
                )}
                {...props}
            />
        );
    }
);

Button.displayName = "Button";

export { Button };
