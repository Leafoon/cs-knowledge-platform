import { getModules } from "@/lib/content-loader";
import { ModuleGrid } from "@/components/knowledge/ModuleGrid";

export default function HomePage() {
    const modules = getModules();

    return (
        <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
            {/* Decorative Elements - Much Larger */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-10 left-0 w-96 h-96 bg-accent-primary opacity-10 rounded-full blur-3xl animate-float" />
                <div className="absolute bottom-10 right-0 w-[500px] h-[500px] bg-accent-secondary opacity-10 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-accent-tertiary opacity-5 rounded-full blur-3xl animate-float" style={{ animationDelay: '4s' }} />
            </div>

            <div className="container mx-auto px-6 py-24 relative z-10">
                {/* Hero Section */}
                <section className="hero text-center mb-24 animate-slide-up">
                    {/* Badge - More Prominent with Pulse */}
                    <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-bg-elevated/80 backdrop-blur-md border-2 border-accent-primary/30 mb-10 shadow-lg animate-pulse">
                        <span className="relative flex h-3 w-3">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-primary opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-accent-primary shadow-lg"></span>
                        </span>
                        <span className="text-base font-bold text-accent-primary">专业级知识学习平台</span>
                    </div>

                    {/* Main Title - Much Larger */}
                    <h1 className="text-7xl md:text-8xl lg:text-9xl font-black mb-8 leading-tight">
                        <span className="bg-gradient-to-r from-accent-primary via-accent-secondary to-accent-tertiary bg-clip-text text-transparent animate-gradient bg-[length:200%_auto] drop-shadow-2xl">
                            CS Knowledge
                        </span>
                        <br />
                        <span className="text-text-primary drop-shadow-lg">Platform</span>
                    </h1>

                    {/* Subtitle - Larger */}
                    <p className="text-2xl md:text-3xl text-text-secondary mb-8 max-w-4xl mx-auto font-medium">
                        系统性学习计算机知识，打造专业技术能力
                    </p>

                    {/* Description */}
                    <p className="text-lg md:text-xl text-text-tertiary max-w-3xl mx-auto leading-relaxed mb-12">
                        涵盖操作系统、编程语言、深度学习等计算机核心领域<br className="hidden md:block" />
                        参考一线互联网大厂标准，构建完整知识体系
                    </p>

                    {/* Stats - Larger and More Prominent */}
                    <div className="flex items-center justify-center gap-12 mt-16">
                        <div className="text-center group cursor-default">
                            <div className="text-5xl font-black bg-gradient-to-r from-accent-primary to-accent-secondary bg-clip-text text-transparent mb-2 group-hover:scale-110 transition-transform">
                                {modules.length}
                            </div>
                            <div className="text-base text-text-tertiary font-semibold">知识模块</div>
                        </div>
                        <div className="w-px h-16 bg-gradient-to-b from-transparent via-border-strong to-transparent"></div>
                        <div className="text-center group cursor-default">
                            <div className="text-5xl font-black bg-gradient-to-r from-accent-secondary to-accent-tertiary bg-clip-text text-transparent mb-2 group-hover:scale-110 transition-transform">
                                ∞
                            </div>
                            <div className="text-base text-text-tertiary font-semibold">持续更新</div>
                        </div>
                        <div className="w-px h-16 bg-gradient-to-b from-transparent via-border-strong to-transparent"></div>
                        <div className="text-center group cursor-default">
                            <div className="text-5xl font-black bg-gradient-to-r from-accent-tertiary to-accent-primary bg-clip-text text-transparent mb-2 group-hover:scale-110 transition-transform">
                                100%
                            </div>
                            <div className="text-base text-text-tertiary font-semibold">免费开放</div>
                        </div>
                    </div>
                </section>

                {/* Module Grid */}
                <section className="modules-section">
                    <div className="flex items-center justify-between mb-12">
                        <div>
                            <h2 className="text-4xl font-black text-text-primary mb-3 bg-gradient-to-r from-text-primary to-text-secondary bg-clip-text">
                                探索知识模块
                            </h2>
                            <p className="text-lg text-text-secondary">
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
