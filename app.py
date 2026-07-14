from ehs_spatial.app import APP_CSS, build_app


def main() -> None:
    build_app().launch(css=APP_CSS, footer_links=[])


if __name__ == "__main__":
    main()
