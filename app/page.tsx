import { getModules } from "@/lib/content-loader";
import { HomePageClient } from "@/components/home/HomePageClient";

export default function HomePage() {
    const modules = getModules();

    return <HomePageClient modules={modules} />;
}
