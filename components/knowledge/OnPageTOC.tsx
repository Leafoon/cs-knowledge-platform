'use client'

import { useEffect, useMemo, useState } from 'react'
import { useScrollSpy } from '@/hooks/useScrollSpy'

interface OnPageTOCProps {
  className?: string
}

interface Heading {
  id: string
  text: string
  level: number
}

export function OnPageTOC({ className = '' }: OnPageTOCProps) {
  const [headings, setHeadings] = useState<Heading[]>([])
  // Memoize the ids array so useScrollSpy's effect doesn't re-run on every render
  const headingIds = useMemo(() => headings.map(h => h.id), [headings])
  const activeId = useScrollSpy(headingIds)

  useEffect(() => {
    const scan = () => {
      const elements = Array.from(
        document.querySelectorAll('.prose-content h2, .prose-content h3')
      ).filter((el) => (el as HTMLElement).id)

      const headingData: Heading[] = elements.map((element) => ({
        id: (element as HTMLElement).id,
        text: element.textContent || '',
        level: parseInt(element.tagName.substring(1)),
      }))

      setHeadings(headingData)
    }

    // Initial scan
    scan()

    // Re-scan whenever the prose-content subtree changes
    // (triggered by router.refresh() replacing RSC payload)
    const container = document.querySelector('.prose-content')
    if (!container) return

    const observer = new MutationObserver(() => {
      // Wait for ContentRenderer's useEffect to assign IDs to headings first,
      // then re-scan. rAF ensures we're after paint; setTimeout gives extra buffer.
      requestAnimationFrame(() => setTimeout(scan, 80))
    })

    observer.observe(container, {
      childList: true,
      subtree: true,
    })

    return () => observer.disconnect()
  }, [])

  if (headings.length === 0) {
    return null
  }

  return (
    <nav className={`space-y-1 ${className}`}>
      <div className="text-xs font-semibold text-text-tertiary uppercase tracking-wide mb-4">
        本页内容
      </div>
      <ul className="space-y-2 text-sm">
        {headings.map((heading) => {
          const isActive = activeId === heading.id
          const isH3 = heading.level === 3

          return (
            <li key={heading.id}>
              <a
                href={`#${heading.id}`}
                onClick={(e) => {
                  e.preventDefault()
                  document.getElementById(heading.id)?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                  })
                }}
                className={`
                  block py-1 transition-colors border-l-2 -ml-px
                  ${isH3 ? 'pl-4' : 'pl-2'}
                  ${
                    isActive
                      ? 'border-accent-primary text-accent-primary font-medium'
                      : 'border-transparent text-text-tertiary hover:text-text-secondary hover:border-border-subtle'
                  }
                `}
              >
                {heading.text}
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
