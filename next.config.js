/** @type {import('next').NextConfig} */
const nextConfig = {
    // output: 'export', // Disabled for development, enable for production build
    images: {
        unoptimized: true,
    },
    reactStrictMode: true,
    trailingSlash: true,
    async headers() {
        return [
            {
                // 对所有章节页面禁用客户端 Router Cache，
                // 防止 Next.js 14 用旧 RSC Payload 覆盖最新服务端内容
                source: '/:module/:chapter/',
                headers: [
                    { key: 'Cache-Control', value: 'no-store' },
                ],
            },
        ]
    },
}

module.exports = nextConfig
