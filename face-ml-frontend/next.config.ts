import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",                   
        destination: "http://localhost:3001/:path*", 
      },
      {
        source: "/uploads/:path*",              
        destination: "http://localhost:3001/uploads/:path*", // proxy ảnh
      },
    ];
  },
};

export default nextConfig;
