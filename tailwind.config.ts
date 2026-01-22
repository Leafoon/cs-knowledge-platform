import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    darkMode: "class",
    theme: {
        extend: {
            colors: {
                'bg-base': 'var(--bg-base)',
                'bg-elevated': 'var(--bg-elevated)',
                'bg-overlay': 'var(--bg-overlay)',
                'text-primary': 'var(--text-primary)',
                'text-secondary': 'var(--text-secondary)',
                'text-tertiary': 'var(--text-tertiary)',
                'border-subtle': 'var(--border-subtle)',
                'border-strong': 'var(--border-strong)',
                'accent-primary': 'var(--accent-primary)',
                'accent-secondary': 'var(--accent-secondary)',
            },
            fontFamily: {
                sans: [
                    "Inter",
                    "-apple-system",
                    "BlinkMacSystemFont",
                    "SF Pro SC",
                    "PingFang SC",
                    "Microsoft YaHei",
                    "sans-serif",
                ],
                mono: [
                    "JetBrains Mono",
                    "Fira Code",
                    "SF Mono",
                    "Consolas",
                    "monospace",
                ],
            },
            spacing: {
                '18': '4.5rem',
                '88': '22rem',
            },
            borderRadius: {
                'sm': '6px',
                'md': '12px',
                'lg': '16px',
            },
            boxShadow: {
                'soft': '0 1px 3px rgba(0, 0, 0, 0.05), 0 4px 16px rgba(0, 0, 0, 0.03)',
                'soft-lg': '0 4px 16px rgba(0, 0, 0, 0.08), 0 12px 32px rgba(0, 0, 0, 0.06)',
                'premium': '0 2px 8px rgba(0, 0, 0, 0.04), 0 8px 24px rgba(0, 0, 0, 0.08), 0 16px 48px rgba(0, 0, 0, 0.12)',
                'glow': '0 0 20px rgba(102, 126, 234, 0.15), 0 0 40px rgba(102, 126, 234, 0.1)',
            },
            animation: {
                'gradient': 'gradient 8s linear infinite',
                'gradient-x': 'gradient-x 3s ease infinite',
                'float': 'float 6s ease-in-out infinite',
                'slide-up': 'slide-up 0.5s ease-out',
                'scale-in': 'scale-in 0.3s ease-out',
                'fade-in': 'fade-in 0.4s ease-out',
                'zoom-in': 'zoom-in 0.3s ease-out',
            },
            keyframes: {
                gradient: {
                    '0%, 100%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                },
                'gradient-x': {
                    '0%, 100%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                },
                float: {
                    '0%, 100%': { transform: 'translateY(0px)' },
                    '50%': { transform: 'translateY(-10px)' },
                },
                'slide-up': {
                    '0%': { opacity: '0', transform: 'translateY(20px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                'scale-in': {
                    '0%': { opacity: '0', transform: 'scale(0.95)' },
                    '100%': { opacity: '1', transform: 'scale(1)' },
                },
                'fade-in': {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                'zoom-in': {
                    '0%': { opacity: '0', transform: 'scale(0.9)' },
                    '100%': { opacity: '1', transform: 'scale(1)' },
                },
            },
            backdropBlur: {
                xs: '2px',
            },
        },
    },
    plugins: [],
};

export default config;
