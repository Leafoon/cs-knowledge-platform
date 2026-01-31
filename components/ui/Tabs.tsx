"use client"

import * as React from "react"
import { motion } from "framer-motion"
import clsx from "clsx"

// Context to share state between components
const TabsContext = React.createContext<{
    value: string
    onChange: (value: string) => void
} | null>(null)

export function Tabs({
    defaultValue,
    value,
    onValueChange,
    children,
    className,
}: {
    defaultValue?: string
    value?: string
    onValueChange?: (value: string) => void
    children: React.ReactNode
    className?: string
}) {
    const [internalValue, setInternalValue] = React.useState(defaultValue || "")

    const currentValue = value !== undefined ? value : internalValue

    const handleChange = (newValue: string) => {
        setInternalValue(newValue)
        onValueChange?.(newValue)
    }

    return (
        <TabsContext.Provider value={{ value: currentValue, onChange: handleChange }}>
            <div className={className}>{children}</div>
        </TabsContext.Provider>
    )
}

export function TabsList({
    children,
    className,
}: {
    children: React.ReactNode
    className?: string
}) {
    return (
        <div
            className={clsx(
                "inline-flex h-10 items-center justify-center rounded-md bg-slate-100 p-1 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
                className
            )}
        >
            {children}
        </div>
    )
}

export function TabsTrigger({
    value,
    children,
    className,
    ...props
}: {
    value: string
    children: React.ReactNode
    className?: string
    disabled?: boolean
}) {
    const context = React.useContext(TabsContext)
    if (!context) throw new Error("TabsTrigger must be used within Tabs")

    const isActive = context.value === value

    return (
        <button
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => context.onChange(value)}
            className={clsx(
                "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 dark:ring-offset-slate-950 dark:focus-visible:ring-slate-300",
                isActive
                    ? "bg-white text-slate-950 shadow-sm dark:bg-slate-950 dark:text-slate-50"
                    : "hover:bg-slate-200/50 dark:hover:bg-slate-700/50",
                className
            )}
            {...props}
        >
            {children}
        </button>
    )
}

export function TabsContent({
    value,
    children,
    className,
    ...props
}: {
    value: string
    children: React.ReactNode
    className?: string
}) {
    const context = React.useContext(TabsContext)
    if (!context) throw new Error("TabsContent must be used within Tabs")

    if (context.value !== value) return null

    return (
        <div
            role="tabpanel"
            className={clsx(
                "mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 dark:ring-offset-slate-950 dark:focus-visible:ring-slate-300",
                className
            )}
            {...props}
        >
            {children}
        </div>
    )
}
