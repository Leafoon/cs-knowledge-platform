import { ReactNode } from "react";
import { getModuleContent } from "@/lib/content-loader";
import { ModuleLayoutClient } from "./ModuleLayoutClient";

export default async function ModuleLayout({
    children,
    params,
}: {
    children: ReactNode;
    params: { module: string };
}) {
    const { toc } = await getModuleContent(params.module);

    return <ModuleLayoutClient toc={toc}>{children}</ModuleLayoutClient>;
}
