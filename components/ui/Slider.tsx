"use client"

import * as React from "react"
import clsx from "clsx"

interface SliderProps {
    value: number[]
    min?: number
    max?: number
    step?: number
    onValueChange?: (value: number[]) => void
    className?: string
    disabled?: boolean
}

export function Slider({
    value,
    min = 0,
    max = 100,
    step = 1,
    onValueChange,
    className,
    disabled = false,
    ...props
}: SliderProps) {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onValueChange?.([parseFloat(e.target.value)])
    }

    return (
        <div className={clsx("relative flex w-full touch-none select-none items-center", className)} {...props}>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value[0]}
                onChange={handleChange}
                disabled={disabled}
                className={clsx(
                    "h-2 w-full appearance-none cursor-pointer rounded-full bg-slate-200 dark:bg-slate-800",
                    "accent-blue-600 dark:accent-blue-500", // Basic browser native styling
                    disabled && "cursor-not-allowed opacity-50"
                )}
            />
        </div>
    )
}
