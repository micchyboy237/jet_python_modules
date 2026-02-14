from jet.libs.smolagents.agents.html_summary_multi_agent import (
    ScalableHTMLMultiAgentSummarizer,
)

if __name__ == "__main__":
    # Example HTML input (could come from requests, file, DB, etc.)
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Example Product Page</title>
            <meta name="description" content="A demo product page for testing.">
        </head>
        <body>
            <header>
                <h1 class="product-title">UltraWidget 3000</h1>
                <p class="subtitle">The most powerful widget ever built.</p>
            </header>

            <section id="pricing">
                <h2>Pricing</h2>
                <table>
                    <tr>
                        <th>Plan</th>
                        <th>Price</th>
                    </tr>
                    <tr>
                        <td>Basic</td>
                        <td>$19/month</td>
                    </tr>
                    <tr>
                        <td>Pro</td>
                        <td>$49/month</td>
                    </tr>
                </table>
            </section>

            <section id="features">
                <h2>Features</h2>
                <ul>
                    <li>Fast performance</li>
                    <li>Cloud sync</li>
                    <li>Offline mode</li>
                </ul>
            </section>

            <footer>
                <a href="https://example.com/contact">Contact Us</a>
            </footer>
        </body>
    </html>
    """

    # Initialize summarizer
    summarizer = ScalableHTMLMultiAgentSummarizer()

    # Run summarization
    summary = summarizer.summarize(html_content)

    print("\n\n=== RAW SUMMARY OUTPUT ===\n")
    print(summary)
