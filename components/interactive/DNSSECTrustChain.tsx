"use client";
import { useState } from "react";

interface DSRecord {
  zone: string;
  keyTag: number;
  algorithm: string;
  digestType: string;
  digest: string;
  verified: boolean;
}

export function DNSSECTrustChain() {
  const [chain] = useState<DSRecord[]>([
    { zone: ".", keyTag: 20326, algorithm: "ECDSAP256SHA256", digestType: "SHA-256", digest: "E06D44B80B8F1D39A95C0B0D7C65D08458E88040...", verified: false },
    { zone: "com.", keyTag: 30909, algorithm: "ECDSAP256SHA256", digestType: "SHA-256", digest: "42CE1D028EF0B3D0E5A8B1DC2CA90D8C4A3B...", verified: false },
    { zone: "example.com.", keyTag: 41567, algorithm: "ECDSAP256SHA256", digestType: "SHA-256", digest: "A8B7F2C3D4E5F6A7B8C9D0E1F2A3B4C5...", verified: false },
    { zone: "www.example.com.", keyTag: 55432, algorithm: "ECDSAP256SHA256", digestType: "SHA-256", digest: "1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D...", verified: false },
  ]);
  const [verified, setVerified] = useState<Set<number>>(new Set());

  const verifyStep = (idx: number) => {
    if (idx > 0 && !verified.has(idx - 1)) return;
    setVerified((prev) => new Set([...prev, idx]));
  };

  const verifyAll = () => {
    for (let i = 0; i < chain.length; i++) {
      setTimeout(() => setVerified((prev) => new Set([...prev, i])), i * 500);
    }
  };

  const verifiedCount = verified.size;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNSSEC 信任链</h3>
      <p className="text-sm text-text-secondary mb-4">从根区信任锚(Trust Anchor)开始，逐级验证DS→DNSKEY→RRSIG签名链。</p>
      <div className="space-y-3 mb-4">
        {chain.map((r, i) => (
          <div key={i} className={`p-3 rounded border transition-all duration-500 ${verified.has(i) ? "border-green-400 bg-green-50 dark:bg-green-900/10" : i === 0 || verified.has(i - 1) ? "border-blue-300" : "border-border-subtle opacity-40"}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-sm text-text-primary font-semibold">{r.zone}</span>
              {verified.has(i) ? (
                <span className="text-xs text-green-600 bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded">✓ 已验证</span>
              ) : (
                <button onClick={() => verifyStep(i)} disabled={i > 0 && !verified.has(i - 1)}
                  className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:dark:bg-gray-700 text-white rounded disabled:cursor-not-allowed">
                  验证
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-1 text-xs text-text-secondary">
              <div>Key Tag: <span className="font-mono text-text-primary">{r.keyTag}</span></div>
              <div>算法: <span className="font-mono text-text-primary">{r.algorithm}</span></div>
              <div className="col-span-2">Digest: <span className="font-mono text-text-primary break-all">{r.digest}</span></div>
            </div>
            {i < chain.length - 1 && <div className="text-center text-text-secondary text-xs mt-2">↓ DS记录传递信任 ↓</div>}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">已验证</div>
          <div className="font-bold text-text-primary">{verifiedCount}/{chain.length}</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">验证算法</div>
          <div className="font-bold text-text-primary text-xs">ECDSA-P256</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">摘要算法</div>
          <div className="font-bold text-text-primary text-xs">SHA-256</div>
        </div>
      </div>
      <button onClick={verifyAll} className="w-full py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm font-medium">
        逐步验证全链
      </button>
    </div>
  );
}
export default DNSSECTrustChain;
