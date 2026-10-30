"""Autonomous-CAE command-line entry point."""

import typer

app = typer.Typer(add_completion=False, help="Autonomous-CAE: AI-instructed parametric CAD pipeline.")


@app.command()
def run() -> None:
    """Run the pipeline (implemented in later build steps)."""
    typer.echo("Autonomous-CAE scaffold is installed. Pipeline not implemented yet.")


if __name__ == "__main__":
    app()
