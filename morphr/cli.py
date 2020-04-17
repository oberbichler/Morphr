from morphr.configuration import Configuration
from morphr import __version__
import click


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name='morphr')
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is not None:
        return
    configuration = Configuration.load('configuration.json')
    configuration.run()

if __name__ == '__main__':
    cli()
