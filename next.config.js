/** @type {import('next').NextConfig} */
const nextConfig = {
    // output: 'export', // Disabled for development, enable for production build
    images: {
        unoptimized: true,
    },
    reactStrictMode: true,
    trailingSlash: true,
}

module.exports = nextConfig
