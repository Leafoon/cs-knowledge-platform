'use client'

import { useEffect, useState } from 'react'
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
  const activeId = useScrollSpy(headings.map(h => h.id))

  useEffect(() => {
    // Extract H2 and H3 headings from the page
    const elements = Array.from(
      document.querySelectorAll('.prose-content h2, .prose-content h3')
    )

    const headingData: Heading[] = elements.map((element) => ({
      id: element.id,
      text: element.textContent || '',
      level: parseInt(element.tagName.substring(1)),
    }))

    setHeadings(headingData)
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
