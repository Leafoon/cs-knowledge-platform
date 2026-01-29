import { visit } from 'unist-util-visit';
import type { Root, Element } from 'hast';

/**
 * Rehype plugin to transform <InteractiveComponent name="XXX" /> tags
 * into <div data-component="XXX"></div> for client-side rendering
 */
export function rehypeInteractiveComponents() {
  return (tree: Root) => {
    visit(tree, 'element', (node: Element) => {
      if (node.tagName === 'interactivecomponent') {
        // Get the component name from the "name" attribute
        const nameAttr = node.properties?.name as string | undefined;
        
        if (nameAttr) {
          // Transform to div with data-component attribute
          node.tagName = 'div';
          node.properties = {
            'data-component': nameAttr,
            className: 'interactive-component-marker'
          };
          node.children = []; // Clear any children
        }
      }
    });
  };
}
