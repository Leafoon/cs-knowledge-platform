const { getSingleChapterContent } = require('./lib/content-loader.ts');

async function test() {
    const result = await getSingleChapterContent('reinforcement-learning', '00-overview');

    // Search for component markers in the HTML
    const componentMatches = result.html.match(/<div[^>]*data-component[^>]*>/g);
    const interactiveMatches = result.html.match(/<interactivecomponent[^>]*>/gi);

    console.log('=== Component markers found ===');
    if (componentMatches) {
        console.log('data-component divs:', componentMatches.length);
        componentMatches.forEach((match, i) => {
            console.log(`${i + 1}. ${match}`);
        });
    } else {
        console.log('No data-component divs found');
    }

    console.log('\n=== InteractiveComponent tags found ===');
    if (interactiveMatches) {
        console.log('InteractiveComponent tags:', interactiveMatches.length);
        interactiveMatches.forEach((match, i) => {
            console.log(`${i + 1}. ${match}`);
        });
    } else {
        console.log('No InteractiveComponent tags found');
    }

    // Show a snippet around where components should be
    const snippets = result.html.match(/.{0,100}(data-component|InteractiveComponent|AgentEnvironmentLoop).{0,100}/gi);
    if (snippets) {
        console.log('\n=== HTML snippets ===');
        snippets.forEach((snippet, i) => {
            console.log(`${i + 1}. ...${snippet}...`);
        });
    }
}

test().catch(console.error);
