import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";

const inter = Inter({
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
    return (
        <html lang="zh-CN" className={inter.variable}>
            <body className="antialiased flex flex-col min-h-screen">
                <Header />
                <main className="flex-1">
                    {children}
                </main>
                <Footer />
            </body>
        </html>
    );
}
