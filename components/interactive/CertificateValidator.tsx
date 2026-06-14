"use client";
import { useState } from "react";

interface CertField {
  field: string;
  value: string;
  valid: boolean | null;
}

interface Certificate {
  subject: string;
  issuer: string;
  validFrom: string;
  validTo: string;
  serial: string;
  algorithm: string;
  fields: CertField[];
}

const certChain: Certificate[] = [
  {
    subject: "Root CA",
    issuer: "Root CA",
    validFrom: "2020-01-01",
    validTo: "2030-12-31",
    serial: "0x01",
    algorithm: "SHA256-RSA",
    fields: [
      { field: "Subject", value: "CN=Root CA, O=Example Org", valid: null },
      { field: "Issuer", value: "CN=Root CA, O=Example Org (自签名)", valid: null },
      { field: "Validity", value: "2020-01-01 ~ 2030-12-31", valid: null },
      { field: "Public Key", value: "RSA 4096-bit", valid: null },
      { field: "Signature", value: "自签名(Self-Signed)", valid: null },
    ],
  },
  {
    subject: "Intermediate CA",
    issuer: "Root CA",
    validFrom: "2022-03-15",
    validTo: "2027-03-14",
    serial: "0x02",
    algorithm: "SHA256-RSA",
    fields: [
      { field: "Subject", value: "CN=Intermediate CA, O=Example Org", valid: null },
      { field: "Issuer", value: "CN=Root CA, O=Example Org", valid: null },
      { field: "Validity", value: "2022-03-15 ~ 2027-03-14", valid: null },
      { field: "Basic Constraints", value: "CA:TRUE, pathlen:0", valid: null },
      { field: "Key Usage", value: "Certificate Sign, CRL Sign", valid: null },
    ],
  },
  {
    subject: "www.example.com",
    issuer: "Intermediate CA",
    validFrom: "2024-06-01",
    validTo: "2025-05-31",
    serial: "0x03",
    algorithm: "SHA256-ECDSA",
    fields: [
      { field: "Subject", value: "CN=www.example.com, O=Example Inc", valid: null },
      { field: "Issuer", value: "CN=Intermediate CA, O=Example Org", valid: null },
      { field: "Validity", value: "2024-06-01 ~ 2025-05-31", valid: null },
      { field: "Subject Alt Names", value: "www.example.com, example.com", valid: null },
      { field: "Key Usage", value: "Digital Signature, Key Encipherment", valid: null },
    ],
  },
];

export function CertificateValidator() {
  const [verifiedIdx, setVerifiedIdx] = useState<Set<number>>(new Set());
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const verify = (idx: number) => {
    if (idx === 0) {
      setVerifiedIdx((prev) => new Set([...prev, idx]));
      return;
    }
    if (!verifiedIdx.has(idx - 1)) return;
    setVerifiedIdx((prev) => new Set([...prev, idx]));
  };

  const verifyAll = () => {
    certChain.forEach((_, i) => {
      setTimeout(() => setVerifiedIdx((prev) => new Set([...prev, i])), i * 600);
    });
  };

  const isExpired = (cert: Certificate) => {
    const now = new Date();
    return now > new Date(cert.validTo);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">X.509 证书链验证</h3>
      <div className="space-y-3 mb-4">
        {certChain.map((cert, i) => (
          <div key={i} className={`p-3 rounded-lg border-2 transition-all ${verifiedIdx.has(i) ? "border-green-400 bg-green-50 dark:bg-green-900/10" : i === 0 || verifiedIdx.has(i - 1) ? "border-blue-300" : "border-border-subtle opacity-40"}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-0.5 rounded ${i === 0 ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400" : i === 1 ? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400" : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"}`}>
                  {i === 0 ? "Root CA" : i === 1 ? "Intermediate" : "End Entity"}
                </span>
                <span className="text-sm font-bold text-text-primary">{cert.subject}</span>
                {isExpired(cert) && <span className="text-xs text-red-600 bg-red-100 dark:bg-red-900/30 px-1 rounded">已过期</span>}
              </div>
              {verifiedIdx.has(i) ? (
                <span className="text-xs text-green-600 bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded">✓ 已验证</span>
              ) : (
                <button onClick={() => verify(i)} disabled={i > 0 && !verifiedIdx.has(i - 1)}
                  className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:dark:bg-gray-700 text-white rounded disabled:cursor-not-allowed">
                  验证
                </button>
              )}
            </div>
            <button onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
              className="text-xs text-text-secondary hover:text-text-primary">
              {expandedIdx === i ? "收起详情 ▲" : "展开详情 ▼"}
            </button>
            {expandedIdx === i && (
              <div className="mt-2 space-y-1">
                {cert.fields.map((f, j) => (
                  <div key={j} className="flex text-xs">
                    <span className="w-24 text-text-secondary">{f.field}</span>
                    <span className="font-mono text-text-primary">{f.value}</span>
                  </div>
                ))}
              </div>
            )}
            {i < certChain.length - 1 && verifiedIdx.has(i) && (
              <div className="text-center text-xs text-text-secondary mt-2">↓ 签发验证 ↓</div>
            )}
          </div>
        ))}
      </div>
      <button onClick={verifyAll} className="w-full py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm font-medium">
        验证整条链
      </button>
      <p className="text-xs text-text-secondary mt-3">从信任锚(Root CA)开始，逐级验证签名、有效期和用途约束，直到终端实体证书。</p>
    </div>
  );
}
export default CertificateValidator;
