import hrequests


def request_url(url: str, showBrowser: bool = False) -> hrequests.HTML:
    print("Initializing session with browser: %s", showBrowser)
    if not showBrowser:
        session: hrequests.Session = hrequests.Session()
    else:
        session: hrequests.BrowserSession = hrequests.BrowserSession(
            browser='chrome',
            headless=False,
        )
    response: hrequests.Response = session.get(url)

    if response.ok:
        print("SUCCESS")
        html_content = response.html.raw_html.decode('utf-8')
        html_parser = hrequests.HTML(
            session=session, html=html_content)

        session.close()

        return html_parser
    else:
        reason = response.reason
        # throw error
        raise Exception(
            f"Failed attempt - Status Code: {response.status_code}, Reason: {reason}")
