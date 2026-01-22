'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Folder, File, ChevronRight, ChevronDown, Info } from 'lucide-react'

interface FileNode {
  name: string
  type: 'file' | 'folder'
  description: string
  children?: FileNode[]
  size?: string
  important?: boolean
}

export default function ModelRepoStructureExplorer() {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['root']))
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  const repoStructure: FileNode = {
    name: 'bert-base-uncased',
    type: 'folder',
    description: '模型仓库根目录',
    children: [
      {
        name: 'config.json',
        type: 'file',
        description: '模型配置文件（架构参数：层数、隐藏层大小、注意力头数等）',
        size: '571 B',
        important: true
      },
      {
        name: 'pytorch_model.bin',
        type: 'file',
        description: '模型权重文件（PyTorch 格式，包含所有参数）',
        size: '440 MB',
        important: true
      },
      {
        name: 'model.safetensors',
        type: 'file',
        description: '安全张量格式权重（推荐格式，加载更快更安全）',
        size: '440 MB',
        important: true
      },
      {
        name: 'tokenizer_config.json',
        type: 'file',
        description: 'Tokenizer 配置文件（特殊 token、最大长度等）',
        size: '350 B',
        important: true
      },
      {
        name: 'vocab.txt',
        type: 'file',
        description: '词汇表文件（所有 token 列表，WordPiece 算法）',
        size: '232 KB',
        important: true
      },
      {
        name: 'tokenizer.json',
        type: 'file',
        description: 'Fast Tokenizer 文件（Rust 实现，加速 tokenization）',
        size: '466 KB'
      },
      {
        name: 'special_tokens_map.json',
        type: 'file',
        description: '特殊 token 映射（[CLS]、[SEP]、[PAD]、[MASK]）',
        size: '125 B'
      },
      {
        name: 'README.md',
        type: 'file',
        description: '模型说明文档（用途、训练数据、引用等）',
        size: '7 KB'
      },
      {
        name: '.gitattributes',
        type: 'file',
        description: 'Git LFS 配置文件（大文件追踪）',
        size: '1.5 KB'
      }
    ]
  }

  const toggleFolder = (path: string) => {
    const newExpanded = new Set(expandedFolders)
    if (newExpanded.has(path)) {
      newExpanded.delete(path)
    } else {
      newExpanded.add(path)
    }
    setExpandedFolders(newExpanded)
  }

  const renderNode = (node: FileNode, path: string, depth: number = 0) => {
    const isExpanded = expandedFolders.has(path)
    const isSelected = selectedFile === path

    return (
      <div key={path} style={{ marginLeft: `${depth * 20}px` }}>
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: depth * 0.05 }}
          onClick={() => {
            if (node.type === 'folder') {
              toggleFolder(path)
            } else {
              setSelectedFile(path)
            }
          }}
          className={`flex items-center gap-2 py-2 px-3 rounded-lg cursor-pointer transition-all ${
            isSelected
              ? 'bg-blue-100 dark:bg-blue-900/30 border-l-4 border-blue-500'
              : 'hover:bg-slate-100 dark:hover:bg-slate-800'
          }`}
        >
          {node.type === 'folder' ? (
            <>
              {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
              <Folder className="w-5 h-5 text-yellow-500" />
            </>
          ) : (
            <>
              <div className="w-4" />
              <File className={`w-5 h-5 ${
                node.important ? 'text-green-500' : 'text-slate-400'
              }`} />
            </>
          )}
          <span className={`font-mono text-sm ${
            node.important ? 'font-bold text-slate-900 dark:text-white' : 'text-slate-600 dark:text-slate-400'
          }`}>
            {node.name}
          </span>
          {node.size && (
            <span className="text-xs text-slate-500 ml-auto">{node.size}</span>
          )}
          {node.important && (
            <span className="ml-2 px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-xs rounded">必需</span>
          )}
        </motion.div>

        {node.type === 'folder' && isExpanded && node.children && (
          <div className="mt-1">
            {node.children.map((child, idx) =>
              renderNode(child, `${path}/${child.name}`, depth + 1)
            )}
          </div>
        )}
      </div>
    )
  }

  const getSelectedFileInfo = () => {
    if (!selectedFile) return null
    const fileName = selectedFile.split('/').pop()
    const file = repoStructure.children?.find(f => f.name === fileName)
    return file
  }

  const selectedFileInfo = getSelectedFileInfo()

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Folder className="w-5 h-5 text-yellow-500" />
          模型仓库结构浏览器
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          点击文件查看详细说明
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* File Tree */}
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4 max-h-96 overflow-auto">
          {renderNode(repoStructure, 'root')}
        </div>

        {/* File Info */}
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
          {selectedFileInfo ? (
            <motion.div
              key={selectedFile}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex items-start gap-3 mb-4">
                <Info className="w-5 h-5 text-blue-500 mt-1" />
                <div>
                  <h4 className="font-mono font-bold text-lg text-slate-900 dark:text-white mb-2">
                    {selectedFileInfo.name}
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                    {selectedFileInfo.description}
                  </p>
                </div>
              </div>

              {selectedFileInfo.size && (
                <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-700 rounded-lg">
                  <div className="text-xs text-slate-600 dark:text-slate-400">文件大小</div>
                  <div className="text-lg font-bold text-slate-900 dark:text-white mt-1">
                    {selectedFileInfo.size}
                  </div>
                </div>
              )}

              {selectedFileInfo.important && (
                <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                  <div className="text-sm font-semibold text-green-700 dark:text-green-400">
                    ⚠️ 核心文件
                  </div>
                  <div className="text-xs text-green-600 dark:text-green-500 mt-1">
                    模型加载时必需的文件
                  </div>
                </div>
              )}
            </motion.div>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-400">
              <div className="text-center">
                <File className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">点击左侧文件查看详情</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Statistics */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">总文件数</div>
          <div className="text-2xl font-bold text-blue-500 mt-1">
            {repoStructure.children?.length || 0}
          </div>
        </div>
        <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">核心文件</div>
          <div className="text-2xl font-bold text-green-500 mt-1">
            {repoStructure.children?.filter(f => f.important).length || 0}
          </div>
        </div>
        <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">总大小</div>
          <div className="text-2xl font-bold text-purple-500 mt-1">~880MB</div>
        </div>
      </div>
    </div>
  )
}
