from pipdeptree._cli import Options
from pipdeptree._discovery import get_installed_distributions
from pipdeptree._models import PackageDAG
from pipdeptree._render import render


def main() -> None:
    pkgs = get_installed_distributions(local_only=True)  # ← customize as needed
    tree = PackageDAG.from_pkgs(pkgs)

    # Optionally: tree = tree.reverse()

    options = Options(
        output_format="graphviz-dot",  # outputs DOT source to stdout
        warn="silence",
        # Optionally add further args: packages, exclude, etc.
    )

    render(options, tree)


if __name__ == "__main__":
    main()
