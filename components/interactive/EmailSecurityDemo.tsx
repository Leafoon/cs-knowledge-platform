"use client";
import { useState } from "react";

export function EmailSecurityDemo() {
  const [selected, setSelected] = useState<"pgp" | "smime">("pgp");
  const [showEncrypt, setShowEncrypt] = useState(false);
  const [showSign, setShowSign] = useState(false);

  const features = {
    pgp: {
      name: "PGP (Pretty Good Privacy)",
      trust: "Web of Trust (信任网)",
      cert: "自签名证书",
      format: "二进制/ASCII Armor",
      algo: "RSA/DSA + IDEA/3DES/AES",
      desc: "基于信任网模型,用户相互签名建立信任链。适合个人用户。",
    },
    smime: {
      name: "S/MIME (Secure MIME)",
      trust: "CA层次信任 (PKI)",
      cert: "X.509证书",
      format: "MIME附件",
      algo: "RSA + 3DES/AES/RC2",
      desc: "基于CA证书体系,适合企业环境。内置于多数邮件客户端。",
    },
  };

  const f = features[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">邮件安全: PGP vs S/MIME</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => setSelected("pgp")}
          className={`px-4 py-2 rounded text-sm ${selected === "pgp" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>PGP</button>
        <button onClick={() => setSelected("smime")}
          className={`px-4 py-2 rounded text-sm ${selected === "smime" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>S/MIME</button>
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <h4 className="font-semibold text-text-primary mb-2">{f.name}</h4>
        <div className="grid grid-cols-2 gap-2 text-sm text-text-secondary">
          <div><strong>信任模型:</strong> {f.trust}</div>
          <div><strong>证书格式:</strong> {f.cert}</div>
          <div><strong>数据格式:</strong> {f.format}</div>
          <div><strong>算法:</strong> {f.algo}</div>
        </div>
        <p className="text-xs text-text-secondary mt-2">{f.desc}</p>
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-bg-muted rounded-lg p-3">
          <button onClick={() => setShowEncrypt(!showEncrypt)}
            className="text-sm font-semibold text-text-primary mb-2 cursor-pointer hover:text-blue-500">
            {showEncrypt ? "▼" : "▶"} 加密过程
          </button>
          {showEncrypt && (
            <div className="text-xs text-text-secondary space-y-1 mt-2">
              <p>1. 发件人生成随机会话密钥K</p>
              <p>2. 用K加密邮件正文(对称加密)</p>
              <p>3. 用收件人公钥加密K(非对称加密)</p>
              <p>4. 发送: 加密正文 + 加密K</p>
              <p>5. 收件人用私钥解密K,再解密正文</p>
            </div>
          )}
        </div>
        <div className="bg-bg-muted rounded-lg p-3">
          <button onClick={() => setShowSign(!showSign)}
            className="text-sm font-semibold text-text-primary mb-2 cursor-pointer hover:text-blue-500">
            {showSign ? "▼" : "▶"} 签名过程
          </button>
          {showSign && (
            <div className="text-xs text-text-secondary space-y-1 mt-2">
              <p>1. 计算邮件正文的哈希值(摘要)</p>
              <p>2. 用发件人私钥加密摘要(签名)</p>
              <p>3. 发送: 原文 + 签名 + 公钥证书</p>
              <p>4. 收件人用公钥解密签名得摘要</p>
              <p>5. 对比计算的摘要,验证完整性</p>
            </div>
          )}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
        <div className="p-2 bg-bg-muted rounded"><strong>PGP:</strong> 信任网,灵活但需手动管理信任</div>
        <div className="p-2 bg-bg-muted rounded"><strong>S/MIME:</strong> PKI体系,企业级但依赖CA</div>
      </div>
    </div>
  );
}

export default EmailSecurityDemo;
