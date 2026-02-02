from pipdeptree._cli import Options
from pipdeptree._discovery import get_installed_distributions
from pipdeptree._models import PackageDAG
from pipdeptree._render import render


def main() -> None:
    pkgs = get_installed_distributions(local_only=True)
    tree = PackageDAG.from_pkgs(pkgs)

    # Optional: tree = tree.reverse()

    options = Options(
        output_format="text",
        warn="silence",
        # sort=True is default behavior in text render
        # no direct max_depth in Options, but tree can be limited via filtering
    )

    render(options, tree)


if __name__ == "__main__":
    main()
