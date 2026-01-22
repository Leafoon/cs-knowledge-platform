"use client";

import { useEffect, useRef, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { InstructionCycleSimulator, VonNeumannArchitecture, ComputerEvolutionTimeline, SystemLayersVisualization, PythonInterpreterFlow, PythonObjectVisualizer, UnicodeEncodingVisualizer, ListResizingVisualizer, IntegerMemoryLayout, HashTableVisualizer, FunctionCallStackVisualizer, DecoratorExecutionFlow, GeneratorStateVisualizer, ExceptionHierarchyTree } from "@/components/interactive";

interface ContentRendererProps {
    html: string;
    moduleId?: string;
}

const componentMap: Record<string, React.ComponentType> = {
    "InstructionCycleSimulator": InstructionCycleSimulator,
    "VonNeumannArchitecture": VonNeumannArchitecture,
    "ComputerEvolutionTimeline": ComputerEvolutionTimeline,
    "SystemLayersVisualization": SystemLayersVisualization,
    "PythonInterpreterFlow": PythonInterpreterFlow,
    "PythonObjectVisualizer": PythonObjectVisualizer,
    "UnicodeEncodingVisualizer": UnicodeEncodingVisualizer,
    "ListResizingVisualizer": ListResizingVisualizer,
    "IntegerMemoryLayout": IntegerMemoryLayout,
    "HashTableVisualizer": HashTableVisualizer,
    "FunctionCallStackVisualizer": FunctionCallStackVisualizer,
    "DecoratorExecutionFlow": DecoratorExecutionFlow,
    "GeneratorStateVisualizer": GeneratorStateVisualizer,
    "ExceptionHierarchyTree": ExceptionHierarchyTree,
};

export function ContentRenderer({ html, moduleId }: ContentRendererProps) {
    const contentRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!contentRef.current) return;

        // Add IDs to headings for scroll spy
        const headings = contentRef.current.querySelectorAll('h1, h2, h3');
        headings.forEach((heading) => {
            const text = heading.textContent || '';
            const id = text
                .toLowerCase()
                .trim()
                .replace(/\s+/g, '-')
                .replace(/[^\w\-\u4e00-\u9fa5]+/g, '')
                .replace(/\-\-+/g, '-');
            heading.id = id;
        });

        // Inject interactive components
        // Find and replace component markers (div with data-component attribute)
        const markers = contentRef.current.querySelectorAll('div[data-component]');
        markers.forEach((marker) => {
            const componentName = marker.getAttribute('data-component');

            if (componentName) {
                const Component = componentMap[componentName];

                if (Component) {
                    const container = document.createElement('div');
                    container.className = 'interactive-component-container my-8';
                    marker.replaceWith(container);

                    const root = createRoot(container);
                    root.render(<Component />);
                }
            }
        });
    }, [html, moduleId]);

    return (
        <div
            ref={contentRef}
            className="prose-content max-w-none"
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
}
