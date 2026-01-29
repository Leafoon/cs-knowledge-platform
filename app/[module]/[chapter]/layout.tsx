import { ReactNode } from "react";

export default async function ChapterLayout({
    children,
}: {
    children: ReactNode;
}) {
    // 章节页面不使用模块的 layout，避免显示全模块目录
    return <>{children}</>;
}
