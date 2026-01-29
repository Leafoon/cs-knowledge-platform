import { ReactNode } from "react";
import { getModuleContent, getModuleChapters } from "@/lib/content-loader";
import { ModuleLayoutClient } from "./ModuleLayoutClient";

export default async function ModuleLayout({
    children,
    params,
}: {
    children: ReactNode;
    params: { module: string };
}) {
    // 检查是否有章节，如果有章节则不使用侧边栏
    const chapters = getModuleChapters(params.module);
    
    if (chapters.length > 0) {
        // 多章节模块，不使用侧边栏
        return <>{children}</>;
    }
    
    // 单页模块，使用侧边栏
    const { toc } = await getModuleContent(params.module);
    return <ModuleLayoutClient toc={toc}>{children}</ModuleLayoutClient>;
}
