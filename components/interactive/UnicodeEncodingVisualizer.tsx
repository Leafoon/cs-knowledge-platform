"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Encoding = "UTF-8" | "UTF-16" | "UTF-32";

export function UnicodeEncodingVisualizer() {
    const [inputText, setInputText] = useState("Hello 世界 🌍");
    const [selectedEncoding, setSelectedEncoding] = useState<Encoding>("UTF-8");

    // Encode the text to bytes based on the selected encoding
    const encodeToBytes = (text: string, encoding: Encoding): number[] => {
        const encoder = new TextEncoder(); // Always UTF-8

        if (encoding === "UTF-8") {
            return Array.from(encoder.encode(text));
        } else if (encoding === "UTF-16") {
            const bytes: number[] = [];
            for (let i = 0; i < text.length; i++) {
                const code = text.charCodeAt(i);
                bytes.push(code >> 8);    // High byte
                bytes.push(code & 0xFF);  // Low byte
            }
            return bytes;
        } else { // UTF-32
            const codePoints = Array.from(text).map(char => char.codePointAt(0)!);
            const bytes: number[] = [];
            codePoints.forEach(cp => {
                bytes.push((cp >> 24) & 0xFF);
                bytes.push((cp >> 16) & 0xFF);
                bytes.push((cp >> 8) & 0xFF);
                bytes.push(cp & 0xFF);
            });
            return bytes;
        }
    };

    const bytes = encodeToBytes(inputText, selectedEncoding);
    const chars = Array.from(inputText);

    const getEncodingInfo = (encoding: Encoding) => {
        const info = {
            "UTF-8": {
                name: "UTF-8 (可变长度: 1-4字节)",
                color: "bg-blue-500",
                desc: "ASCII兼容，节省空间"
            },
            "UTF-16": {
                name: "UTF-16 (可变长度: 2或4字节)",
                color: "bg-purple-500",
                desc: "常用于Java、Windows"
            },
            "UTF-32": {
                name: "UTF-32 (固定长度: 4字节)",
                color: "bg-green-500",
                desc: "简单但浪费空间"
            },
        };
        return info[encoding];
    };

    const info = getEncodingInfo(selectedEncoding);

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-bg-elevated rounded-xl border border-border-subtle shadow-lg my-8">
            <h3 className="text-2xl font-bold text-center mb-6 text-text-primary">
                Unicode 编码可视化
            </h3>

            {/* Input Section */}
            <div className="mb-6">
                <label className="block text-sm font-semibold text-text-secondary mb-2">
                    输入字符串 (支持中文、Emoji)
                </label>
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    className="w-full px-4 py-3 bg-bg-base border border-border-subtle rounded-lg text-text-primary font-mono text-lg focus:outline-none focus:ring-2 focus:ring-accent-primary"
                    placeholder="输入任意字符..."
                />
            </div>

            {/* Encoding Selector */}
            <div className="flex gap-3 mb-6">
                {(["UTF-8", "UTF-16", "UTF-32"] as Encoding[]).map((enc) => (
                    <button
                        key={enc}
                        onClick={() => setSelectedEncoding(enc)}
                        className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-200 ${selectedEncoding === enc
                                ? `${getEncodingInfo(enc).color} text-white shadow-md scale-105`
                                : "bg-bg-base text-text-secondary hover:bg-bg-elevated border border-border-subtle"
                            }`}
                    >
                        {enc}
                    </button>
                ))}
            </div>

            {/* Encoding Info */}
            <div className={`p-4 rounded-lg mb-6 ${info.color} bg-opacity-10 border border-current border-opacity-20`}>
                <div className="flex items-center justify-between">
                    <div>
                        <div className={`text-lg font-bold`} style={{ color: info.color.replace('bg-', '') }}>
                            {info.name}
                        </div>
                        <div className="text-sm text-text-secondary mt-1">{info.desc}</div>
                    </div>
                    <div className="text-right">
                        <div className="text-2xl font-bold text-text-primary">{bytes.length}</div>
                        <div className="text-xs text-text-tertiary">总字节数</div>
                    </div>
                </div>
            </div>

            {/* Character Breakdown */}
            <div className="space-y-4">
                <h4 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">
                    字符 → 字节映射
                </h4>
                <div className="space-y-3 max-h-96 overflow-y-auto scrollbar-thin">
                    <AnimatePresence mode="popLayout">
                        {chars.map((char, idx) => {
                            const codePoint = char.codePointAt(0)!;
                            const charBytes = encodeToBytes(char, selectedEncoding);

                            return (
                                <motion.div
                                    key={`${char}-${idx}`}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    transition={{ delay: idx * 0.05 }}
                                    className="p-4 bg-bg-base border border-border-subtle rounded-lg hover:border-accent-primary transition-colors"
                                >
                                    <div className="flex items-center gap-4">
                                        {/* Character Display */}
                                        <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-lg flex items-center justify-center text-3xl shadow-md">
                                            {char}
                                        </div>

                                        {/* Character Info */}
                                        <div className="flex-1">
                                            <div className="text-sm text-text-tertiary">
                                                Unicode: <span className="font-mono text-accent-primary">U+{codePoint.toString(16).toUpperCase().padStart(4, '0')}</span>
                                            </div>
                                            <div className="flex gap-2 mt-2 flex-wrap">
                                                {charBytes.map((byte, i) => (
                                                    <motion.div
                                                        key={i}
                                                        initial={{ scale: 0 }}
                                                        animate={{ scale: 1 }}
                                                        transition={{ delay: idx * 0.05 + i * 0.02 }}
                                                        className="px-3 py-1 bg-bg-elevated border border-accent-primary/30 rounded font-mono text-sm text-text-primary"
                                                    >
                                                        0x{byte.toString(16).toUpperCase().padStart(2, '0')}
                                                    </motion.div>
                                                ))}
                                            </div>
                                            <div className="text-xs text-text-tertiary mt-1">
                                                {charBytes.length} 字节
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </AnimatePresence>
                </div>
            </div>

            {/* Complete Byte Stream */}
            <div className="mt-6 p-4 bg-black/80 rounded-lg">
                <div className="text-xs text-green-400 mb-2 font-mono">完整字节流:</div>
                <div className="font-mono text-xs text-green-300 break-all leading-relaxed">
                    {bytes.map((b, i) => (
                        <span key={i} className="mr-1">
                            {b.toString(16).toUpperCase().padStart(2, '0')}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}
