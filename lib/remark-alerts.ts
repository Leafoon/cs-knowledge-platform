import { visit } from "unist-util-visit";

const RE_ALERT = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/;

export function remarkAlerts() {
    return (tree: any) => {
        visit(tree, "blockquote", (node: any) => {
            if (
                node.children &&
                node.children.length > 0 &&
                node.children[0].type === "paragraph" &&
                node.children[0].children &&
                node.children[0].children.length > 0 &&
                node.children[0].children[0].type === "text"
            ) {
                const textNode = node.children[0].children[0];
                const match = textNode.value.match(RE_ALERT);

                if (match) {
                    const type = match[1].toLowerCase();

                    // Remove the [!NOTE] text
                    textNode.value = textNode.value.replace(RE_ALERT, "").trim();

                    // Add data properties for Rehype to render as specific div/class
                    node.data = node.data || {};
                    node.data.hProperties = node.data.hProperties || {};
                    node.data.hProperties.className = [`markdown-alert`, `markdown-alert-${type}`];
                    node.data.hName = "div"; // Change <blockquote> to <div>
                }
            }
        });
    };
}
