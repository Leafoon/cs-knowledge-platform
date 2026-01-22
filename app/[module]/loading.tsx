export default function Loading() {
    return (
        <div className="min-h-screen flex items-center justify-center">
            <div className="flex flex-col items-center gap-4">
                <div className="relative w-16 h-16">
                    <div className="absolute inset-0 border-4 border-accent-primary/20 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-accent-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
                <p className="text-text-secondary text-sm">加载中...</p>
            </div>
        </div>
    );
}
