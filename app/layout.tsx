import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { getModules } from "@/lib/content-loader";
import { SearchProvider } from "@/components/search/SearchContext";
import { CommandMenu } from "@/components/search/CommandMenu";

const outfit = Outfit({
    subsets: ["latin"],
    display: "swap",
    variable: "--font-sans",
});

export const metadata: Metadata = {
    title: "CS Knowledge Platform",
    description: "专业级计算机知识学习平台",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    const modules = getModules();

    return (
        <html lang="zh-CN" className={outfit.variable} suppressHydrationWarning>
            <body className="antialiased flex flex-col min-h-screen" suppressHydrationWarning>
                <SearchProvider>
                    <CommandMenu modules={modules} />
                    <Header />
                    <main className="flex-1">
                        {children}
                    </main>
                    <Footer />
                </SearchProvider>
            </body>
        </html>
    );
}
