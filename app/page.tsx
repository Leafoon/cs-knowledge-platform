import { getModules } from "@/lib/content-loader";
import { ModuleGrid } from "@/components/knowledge/ModuleGrid";

export default function HomePage() {
    const modules = getModules();

    return (
        <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
            {/* Decorative Elements - Optimized for Performance */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-10 left-0 w-96 h-96 bg-accent-primary opacity-5 rounded-full" />
                <div className="absolute bottom-10 right-0 w-[400px] h-[400px] bg-accent-secondary opacity-5 rounded-full" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-accent-tertiary opacity-3 rounded-full" />
            </div>

            <div className="container mx-auto px-4 sm:px-6 py-12 sm:py-16 md:py-20 lg:py-24 relative z-10">
                {/* Hero Section */}
                <section className="hero text-center mb-16 sm:mb-20 md:mb-24 animate-slide-up">
                    {/* Badge - Optimized Performance */}
                    <div className="inline-flex items-center gap-2 sm:gap-3 px-4 sm:px-6 py-2 sm:py-3 rounded-full bg-bg-elevated border-2 border-accent-primary/30 mb-6 sm:mb-8 md:mb-10 shadow-lg">
                        <span className="relative flex h-2 w-2 sm:h-3 sm:w-3">
                            <span className="relative inline-flex rounded-full h-2 w-2 sm:h-3 sm:w-3 bg-accent-primary shadow-lg"></span>
                        </span>
                        <span className="text-sm sm:text-base font-bold text-accent-primary">专业级知识学习平台</span>
                    </div>

                    {/* Main Title - Responsive Sizes */}
                    <h1 className="text-4xl sm:text-5xl md:text-7xl lg:text-8xl xl:text-9xl font-black mb-6 sm:mb-8 leading-tight px-2">
                        <span className="bg-gradient-to-r from-accent-primary via-accent-secondary to-accent-tertiary bg-clip-text text-transparent">
                            CS Knowledge
                        </span>
                        <br />
                        <span className="text-text-primary">Platform</span>
                    </h1>

                    {/* Subtitle - Responsive */}
                    <p className="text-base sm:text-xl md:text-2xl lg:text-3xl text-text-secondary mb-6 sm:mb-8 max-w-4xl mx-auto font-medium px-4">
                        系统性学习计算机知识，打造专业技术能力
                    </p>

                    {/* Description */}
                    <p className="text-sm sm:text-base md:text-lg lg:text-xl text-text-tertiary max-w-3xl mx-auto leading-relaxed mb-8 sm:mb-10 md:mb-12 px-4">
                        涵盖操作系统、编程语言、深度学习等计算机核心领域<br className="hidden md:block" />
                        参考一线互联网大厂标准，构建完整知识体系
                    </p>

                    {/* Stats - Responsive Grid */}
                    <div className="flex items-center justify-center gap-6 sm:gap-8 md:gap-12 mt-8 sm:mt-12 md:mt-16 flex-wrap sm:flex-nowrap">
                        <div className="text-center group cursor-default">
                            <div className="text-3xl sm:text-4xl md:text-5xl font-black bg-gradient-to-r from-accent-primary to-accent-secondary bg-clip-text text-transparent mb-1 sm:mb-2 group-hover:scale-110 transition-transform">
                                {modules.length}
                            </div>
                            <div className="text-xs sm:text-sm md:text-base text-text-tertiary font-semibold">知识模块</div>
                        </div>
                        <div className="w-px h-12 sm:h-14 md:h-16 bg-gradient-to-b from-transparent via-border-strong to-transparent"></div>
                        <div className="text-center group cursor-default">
                            <div className="text-3xl sm:text-4xl md:text-5xl font-black bg-gradient-to-r from-accent-secondary to-accent-tertiary bg-clip-text text-transparent mb-1 sm:mb-2 group-hover:scale-110 transition-transform">
                                ∞
                            </div>
                            <div className="text-xs sm:text-sm md:text-base text-text-tertiary font-semibold">持续更新</div>
                        </div>
                        <div className="w-px h-12 sm:h-14 md:h-16 bg-gradient-to-b from-transparent via-border-strong to-transparent"></div>
                        <div className="text-center group cursor-default">
                            <div className="text-3xl sm:text-4xl md:text-5xl font-black bg-gradient-to-r from-accent-tertiary to-accent-primary bg-clip-text text-transparent mb-1 sm:mb-2 group-hover:scale-110 transition-transform">
                                100%
                            </div>
                            <div className="text-xs sm:text-sm md:text-base text-text-tertiary font-semibold">免费开放</div>
                        </div>
                    </div>
                </section>

                {/* Module Grid */}
                <section className="modules-section">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 sm:mb-10 md:mb-12 px-2 sm:px-0">
                        <div>
                            <h2 className="text-2xl sm:text-3xl md:text-4xl font-black text-text-primary mb-2 sm:mb-3 bg-gradient-to-r from-text-primary to-text-secondary bg-clip-text">
                                探索知识模块
                            </h2>
                            <p className="text-sm sm:text-base md:text-lg text-text-secondary">
                                选择感兴趣的领域，开始你的学习之旅
                            </p>
                        </div>
                    </div>
                    <ModuleGrid modules={modules} />
                </section>
            </div>
        </div>
    );
}
