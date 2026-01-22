import Link from "next/link";

export function Footer() {
    return (
        <footer className="border-t border-border-subtle bg-bg-elevated mt-auto">
            <div className="container mx-auto px-6 py-8">
                <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="text-sm text-text-tertiary">
                        © 2026 CS Knowledge Platform. 专业级计算机知识学习平台
                    </div>

                    <div className="flex items-center space-x-6 text-sm">
                        <Link
                            href="#"
                            className="text-text-secondary hover:text-accent-primary transition-colors"
                        >
                            关于
                        </Link>
                        <Link
                            href="#"
                            className="text-text-secondary hover:text-accent-primary transition-colors"
                        >
                            反馈
                        </Link>
                        <a
                            href="https://github.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-text-secondary hover:text-accent-primary transition-colors"
                        >
                            GitHub
                        </a>
                    </div>
                </div>
            </div>
        </footer>
    );
}
