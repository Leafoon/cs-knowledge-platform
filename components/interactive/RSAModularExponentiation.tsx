"use client";
import { useState, useMemo } from "react";

function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  base = base % mod;
  while (exp > 0n) {
    if (exp % 2n === 1n) result = (result * base) % mod;
    exp = exp / 2n;
    base = (base * base) % mod;
  }
  return result;
}

export function RSAModularExponentiation() {
  const [p, setP] = useState(61);
  const [q, setQ] = useState(53);
  const [e, setE] = useState(17);
  const [plaintext, setPlaintext] = useState("42");
  const [showSteps, setShowSteps] = useState(false);

  const rsa = useMemo(() => {
    const n = BigInt(p) * BigInt(q);
    const phi = BigInt(p - 1) * BigInt(q - 1);
    const eB = BigInt(e);
    let d = 1n;
    for (let k = 1n; k < 100000n; k++) {
      if ((k * phi + 1n) % eB === 0n) {
        d = (k * phi + 1n) / eB;
        break;
      }
    }
    const m = BigInt(plaintext);
    const c = modPow(m, eB, n);
    const decrypted = modPow(c, d, n);
    return { n, phi, d, c, decrypted, valid: decrypted === m };
  }, [p, q, e, plaintext]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">RSA模幂运算演示</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          p (素数): <input type="number" value={p} onChange={(e) => setP(+e.target.value)} className="w-full mt-1 px-2 py-1 rounded bg-bg-tertiary border border-border-subtle text-text-primary font-mono text-sm" />
        </label>
        <label className="text-xs text-text-secondary">
          q (素数): <input type="number" value={q} onChange={(e) => setQ(+e.target.value)} className="w-full mt-1 px-2 py-1 rounded bg-bg-tertiary border border-border-subtle text-text-primary font-mono text-sm" />
        </label>
        <label className="text-xs text-text-secondary">
          e (公钥指数): <input type="number" value={e} onChange={(e) => setE(+e.target.value)} className="w-full mt-1 px-2 py-1 rounded bg-bg-tertiary border border-border-subtle text-text-primary font-mono text-sm" />
        </label>
      </div>
      <label className="text-xs text-text-secondary block mb-4">
        明文 M: <input type="number" value={plaintext} onChange={(e) => setPlaintext(e.target.value)} className="w-32 ml-2 px-2 py-1 rounded bg-bg-tertiary border border-border-subtle text-text-primary font-mono text-sm" />
      </label>
      <div className="space-y-2 mb-4">
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-xs">
          <span className="text-text-tertiary w-20">n = p×q</span>
          <span className="font-mono text-text-primary">{rsa.n.toString()}</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-xs">
          <span className="text-text-tertiary w-20">φ(n)</span>
          <span className="font-mono text-text-primary">{rsa.phi.toString()}</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-xs">
          <span className="text-text-tertiary w-20">d (私钥)</span>
          <span className="font-mono text-text-primary">{rsa.d.toString()}</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-sky-500/10 border border-sky-500/30 text-xs">
          <span className="text-sky-600 dark:text-sky-400 w-20">加密: C</span>
          <span className="font-mono text-sky-700 dark:text-sky-300">M^e mod n = {plaintext}^{e} mod {rsa.n.toString()} = {rsa.c.toString()}</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs">
          <span className="text-emerald-600 dark:text-emerald-400 w-20">解密: M</span>
          <span className="font-mono text-emerald-700 dark:text-emerald-300">C^d mod n = {rsa.c.toString()}^{rsa.d.toString()} mod {rsa.n.toString()} = {rsa.decrypted.toString()}</span>
        </div>
      </div>
      <div className={`text-xs px-3 py-2 rounded-lg ${rsa.valid ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-red-500/10 text-red-600 dark:text-red-400"}`}>
        {rsa.valid ? "✓ 解密成功：明文 = 密文解密结果" : "✗ 解密失败：请检查参数（p,q需为素数，e与φ(n)互素）"}
      </div>
      <button onClick={() => setShowSteps(!showSteps)} className="mt-3 text-xs text-sky-600 dark:text-sky-400 hover:underline">
        {showSteps ? "隐藏" : "显示"}模幂运算步骤
      </button>
      {showSteps && (
        <div className="mt-2 rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
          <div>1. 选择两个大素数 p={p}, q={q}</div>
          <div>2. 计算 n = p×q = {rsa.n.toString()}</div>
          <div>3. 计算 φ(n) = (p-1)(q-1) = {rsa.phi.toString()}</div>
          <div>4. 选择 e={e}，满足 gcd(e, φ(n)) = 1</div>
          <div>5. 计算 d = e⁻¹ mod φ(n) = {rsa.d.toString()}</div>
          <div>6. 加密: C = M^e mod n</div>
          <div>7. 解密: M = C^d mod n</div>
        </div>
      )}
    </div>
  );
}
export default RSAModularExponentiation;
